"""
hypothesis1_reg_nrc_strong.py — гипотеза 1 (12-13.09.2026): регуляризация весов (weight_decay)
для nrc_strong@hidden_dim=64 — единственная точка с устойчивым эффектом (TZ_stage8).

Логика (_train_nrc_strong_reg, _calibrate_weight_decay, run_hypothesis1) перенесена
без изменений; predict_nrc_strong/get_spatial переиспользуются из библиотеки.
"""
from pathlib import Path

import numpy as np
import torch

import json

from gi_surrogate.data.batch import SPATIAL_DIM
from gi_surrogate.data.batch import build_batch_dense
from gi_surrogate.data.batch import get_spatial
from gi_surrogate.data.sampling import tile_stratified_sample
from gi_surrogate.data.zeroday_adapter import load_fully_valid_pixel_sequences_zeroday
from gi_surrogate.eval.slices import slice_mse_by_brightness
from gi_surrogate.eval.stats import paired_bootstrap_significance
from gi_surrogate.eval.stats import seed_robustness_report
from gi_surrogate.models.losses import NRCRelativeL2Loss
from gi_surrogate.models.nrc import NRCStyleBaseline
from gi_surrogate.paths import DEFAULT_DATASET
from gi_surrogate.paths import TZ8_KAN_RESULTS as KAN_RESULTS_PATH
from gi_surrogate.training.nrc import NRC_DEVICE
from gi_surrogate.training.nrc import STRONG_BIAS
from gi_surrogate.training.nrc import STRONG_N_HIDDEN_LAYERS
from gi_surrogate.training.nrc import predict_nrc_strong


HIDDEN_DIM = 64          # единственная пара с устойчивым эффектом (TZ_stage8)


N_TRAIN_PIXELS = 90      # как в TZ_stage8/§9 — обязателен для идентичных сплитов


N_SEEDS = 16             # как в TZ_stage8 — для прямой сопоставимости с nrc_kan


EPOCHS = 200


LR = 1e-3


# пре-регистрированная сетка (arXiv 2605.29039 использует weight decay без
# указания точного значения для их задачи — здесь сетка на порядки величины,
# как CURVATURE_LAMBDA в TZ_stage8 §3.4, а не точечный подбор)
WEIGHT_DECAY_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


WD_CALIB_SEEDS = 4       # отдельные сиды 0..3, используются ТОЛЬКО для подбора


                         # weight_decay по val-loss — не пересекаются по цели
                         # с основным 16-сидовым сравнением ниже (те же сиды
                         # 0..3 переиспользуются в основном прогоне для train/
                         # eval split, но НЕ для выбора гиперпараметра)
WD_VAL_FRACTION = 0.2    # доля train-пикселей (из 90), отложенная под val


def _train_nrc_strong_reg(hidden_dim, batch, device, epochs, lr, weight_decay):
    """Копия _train_nrc_strong (run_capacity_sweep_zeroday_strong_nrc.py) с
    единственным отличием — weight_decay в optimizer (Sestak 2605.29039,
    MLP-Reg). Конфиг (5 слоёв, bias=False) идентичен nrc_strong — меняется
    ТОЛЬКО регуляризация, не архитектура, иначе сравнение станет нечестным
    в другую сторону."""
    obs = batch["obs"].to(device)
    cold = batch["cold"].to(device)
    conf = batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    obs_dim = obs.shape[-1]

    spatial = get_spatial(batch, device)
    model = NRCStyleBaseline(
        obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
        staleness_dim=4, spatial_dim=SPATIAL_DIM,
        n_hidden_layers=STRONG_N_HIDDEN_LAYERS, bias=STRONG_BIAS,
    ).to(device)
    u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                      extra_staleness=[disocc, mismatch], spatial=spatial)
    true = batch["true"].to(device)
    loss_fn = NRCRelativeL2Loss(use_confidence_weight=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(u)
        loss = loss_fn(pred, true)
        loss.backward()
        optimizer.step()
    return model


def _calibrate_weight_decay(seqs, hidden_dim, epochs, lr):
    """Подбор weight_decay ДО основного сравнения, по val-loss на
    калибровочных сидах (WD_CALIB_SEEDS) — НЕ по dark-frame MSE и НЕ по
    сравнению с nrc_kan, чтобы избежать утечки исхода гипотезы в выбор
    гиперпараметра (симметрично принципу TZ_stage8 §3.4 для
    CURVATURE_LAMBDA)."""
    loss_fn = NRCRelativeL2Loss(use_confidence_weight=False)
    mean_val_loss = {wd: [] for wd in WEIGHT_DECAY_GRID}
    for seed in range(WD_CALIB_SEEDS):
        rng = np.random.default_rng(seed)
        train_idx = tile_stratified_sample(seqs, N_TRAIN_PIXELS, rng)
        full_batch = build_batch_dense(seqs, train_idx)
        n_val = max(1, int(round(N_TRAIN_PIXELS * WD_VAL_FRACTION)))
        perm = rng.permutation(N_TRAIN_PIXELS)
        val_pos, sub_pos = perm[:n_val], perm[n_val:]
        sub_batch = {k: (v[sub_pos] if torch.is_tensor(v) else v)
                     for k, v in full_batch.items()}
        val_batch = {k: (v[val_pos] if torch.is_tensor(v) else v)
                     for k, v in full_batch.items()}
        for wd in WEIGHT_DECAY_GRID:
            torch.manual_seed(seed)
            m = _train_nrc_strong_reg(hidden_dim, sub_batch, NRC_DEVICE, epochs, lr, wd)
            with torch.no_grad():
                obs = val_batch["obs"].to(NRC_DEVICE)
                cold = val_batch["cold"].to(NRC_DEVICE)
                conf = val_batch["conf"].to(NRC_DEVICE)
                disocc = val_batch["disocclusion_flag"].to(NRC_DEVICE)
                mismatch = val_batch["geometric_mismatch_score"].to(NRC_DEVICE)
                spatial = get_spatial(val_batch, NRC_DEVICE)
                u = NRCStyleBaseline.build_input(
                    obs, cold, conf, use_staleness=True,
                    extra_staleness=[disocc, mismatch], spatial=spatial)
                pred = m(u)
                val_loss = float(loss_fn(pred, val_batch["true"].to(NRC_DEVICE)).item())
            mean_val_loss[wd].append(val_loss)

    avg = {wd: float(np.mean(v)) for wd, v in mean_val_loss.items()}
    print("=== калибровка weight_decay (val-loss, среднее по "
          f"{WD_CALIB_SEEDS} сидам, hidden_dim={hidden_dim}) ===")
    for wd in WEIGHT_DECAY_GRID:
        print(f"  weight_decay={wd:.0e}: val_loss={avg[wd]:.4e}")
    best_wd = min(avg, key=avg.get)
    print(f"  -> выбрано weight_decay={best_wd:.0e}\n")
    return best_wd


def run_hypothesis1(dataset_path=DEFAULT_DATASET, hidden_dim=HIDDEN_DIM,
                     n_seeds=N_SEEDS, n_train_pixels=N_TRAIN_PIXELS,
                     epochs=EPOCHS, lr=LR, weight_decay=None):
    with open(KAN_RESULTS_PATH) as f:
        kan_results = json.load(f)
    if str(hidden_dim) not in kan_results["dark_by"]["nrc_kan"]:
        raise ValueError(f"hidden_dim={hidden_dim} отсутствует в "
                          f"{KAN_RESULTS_PATH} (dark_by.nrc_kan)")
    kan_dark = np.array(kan_results["dark_by"]["nrc_kan"][str(hidden_dim)])
    plain_strong_dark = np.array(kan_results["dark_by"]["nrc_strong"][str(hidden_dim)])
    if len(kan_dark) != n_seeds or len(plain_strong_dark) != n_seeds:
        raise ValueError(
            f"В {KAN_RESULTS_PATH} для hidden_dim={hidden_dim} сохранено "
            f"{len(kan_dark)}/{len(plain_strong_dark)} сидов nrc_kan/nrc_strong, "
            f"ожидалось {n_seeds} — обновите n_seeds или пересчитайте базовый файл.")

    seqs = load_fully_valid_pixel_sequences_zeroday(dataset_path)

    if weight_decay is None:
        weight_decay = _calibrate_weight_decay(seqs, hidden_dim, epochs, lr)

    print(f"=== Гипотеза 1: nrc_strong_reg (weight_decay={weight_decay:.0e}) "
          f"vs nrc_kan (curvature_lambda={kan_results['curvature_lambda']:.0e}), "
          f"dark-frame MSE, hidden_dim={hidden_dim}, n_seeds={n_seeds} ===")
    reg_dark = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        train_idx = tile_stratified_sample(seqs, n_train_pixels, rng)
        train_idx_set = set(train_idx)
        eval_idx = [i for i in range(len(seqs)) if i not in train_idx_set]
        train_batch = build_batch_dense(seqs, train_idx)
        eval_batch = build_batch_dense(seqs, eval_idx)
        true = eval_batch["true"].numpy()
        n_eval = len(eval_idx)
        dark_threshold = float(np.median(true))

        m = _train_nrc_strong_reg(hidden_dim, train_batch, NRC_DEVICE, epochs, lr, weight_decay)
        pred = predict_nrc_strong(m, eval_batch, NRC_DEVICE)
        dk, _br = slice_mse_by_brightness(pred, true, n_eval, dark_threshold)
        reg_dark.append(dk)
        print(f"  seed {seed+1}/{n_seeds}: nrc_strong_reg dark={dk:.3e} "
              f"(nrc_kan[seed]={kan_dark[seed]:.3e}, "
              f"nrc_strong_plain[seed]={plain_strong_dark[seed]:.3e})")

    reg_dark = np.array(reg_dark)
    sig_vs_kan = paired_bootstrap_significance(kan_dark, reg_dark)
    rob_vs_kan = seed_robustness_report(kan_dark, reg_dark)
    sig_vs_plain = paired_bootstrap_significance(reg_dark, plain_strong_dark)
    rob_vs_plain = seed_robustness_report(reg_dark, plain_strong_dark)

    print("\n=== результат ===")
    print(f"nrc_kan vs nrc_strong_reg (dark, hidden_dim={hidden_dim}): "
          f"mean_diff(kan-reg)={sig_vs_kan['mean_diff']:.3e} "
          f"CI=[{sig_vs_kan['ci_low']:.3e},{sig_vs_kan['ci_high']:.3e}] "
          f"significant={sig_vs_kan['significant']} | "
          f"cohens_d={rob_vs_kan['cohens_d']:.2f} -> {rob_vs_kan['recommendation']}")
    print(f"nrc_strong_reg vs nrc_strong_plain (dark, sanity — регуляризация "
          f"сама по себе меняет ли метрику): "
          f"mean_diff(reg-plain)={sig_vs_plain['mean_diff']:.3e} "
          f"CI=[{sig_vs_plain['ci_low']:.3e},{sig_vs_plain['ci_high']:.3e}] "
          f"significant={sig_vs_plain['significant']} | "
          f"cohens_d={rob_vs_plain['cohens_d']:.2f}")

    gap_closed = (not sig_vs_kan["significant"]) or abs(rob_vs_kan["cohens_d"]) < 0.5
    print("\n=== вердикт Гипотезы 1 ===")
    if gap_closed:
        print("  Разрыв KAN vs MLP на тёмных кадрах схлопнулся после регуляризации "
              "nrc_strong (significant=False или |d|<0.5) — Гипотеза 1 ПОДТВЕРЖДЕНА, "
              "эффект объясняется недорегуляризованным baseline'ом, не архитектурой "
              "KAN. Гипотезы 2/3 не требуются для объяснения ЭТОГО эффекта (могут "
              "оставаться релевантными для переносимости на видео отдельно).")
    else:
        print("  Разрыв остался значим и устойчив даже после регуляризации "
              "nrc_strong — Гипотеза 1 ОТКЛОНЕНА для этой задачи. Переходить к "
              "Гипотезе 2 (шум во входных признаках).")

    results = dict(
        hidden_dim=hidden_dim, n_seeds=n_seeds, weight_decay=weight_decay,
        weight_decay_grid=WEIGHT_DECAY_GRID,
        reg_dark=reg_dark.tolist(), kan_dark=kan_dark.tolist(),
        plain_strong_dark=plain_strong_dark.tolist(),
        sig_vs_kan=sig_vs_kan, rob_vs_kan=rob_vs_kan,
        sig_vs_plain=sig_vs_plain, rob_vs_plain=rob_vs_plain,
        gap_closed=bool(gap_closed),
    )
    out_path = Path(dataset_path).parent / "run_hypothesis1_reg_nrc_strong_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                   default=lambda o: bool(o) if isinstance(o, np.bool_) else o)
    print(f"\nСырые результаты сохранены в {out_path}")
    return results
