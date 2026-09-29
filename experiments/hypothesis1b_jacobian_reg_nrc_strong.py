"""
hypothesis1b_jacobian_reg_nrc_strong.py — гипотеза 1b (13.09.2026): якобиан-регуляризация
(градиент выхода по входу) для nrc_strong@hidden_dim=64, альтернатива weight_decay
из гипотезы 1 (см. hypothesis1_reg_nrc_strong.py, HYPOTHESIS1_RESULTS).

Логика перенесена без изменений; predict_nrc_strong/get_spatial/slice_mse_by_brightness
переиспользуются из библиотеки, не дублируются (диф с run_capacity_sweep_zeroday_kan.py
подтвердил идентичность логики, отличия только в докстринге).
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
from gi_surrogate.paths import HYPOTHESIS1_RESULTS as H1_RESULTS_PATH
from gi_surrogate.paths import TZ8_KAN_RESULTS as KAN_RESULTS_PATH
from gi_surrogate.training.nrc import NRC_DEVICE
from gi_surrogate.training.nrc import STRONG_BIAS
from gi_surrogate.training.nrc import STRONG_N_HIDDEN_LAYERS
from gi_surrogate.training.nrc import predict_nrc_strong


HIDDEN_DIM = 64


N_TRAIN_PIXELS = 90


N_SEEDS = 16


EPOCHS = 200


LR = 1e-3


# пре-регистрированная сетка (порядки величины, как WEIGHT_DECAY_GRID и
# CURVATURE_LAMBDA — не точечный подбор под исход)
GRADIENT_LAMBDA_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


GRAD_CALIB_SEEDS = 4


GRAD_VAL_FRACTION = 0.2


def _jacobian_penalty(pred, u):
    """Sum_c ||grad_u pred[...,c]||^2, усреднённая по batch*time — точный
    Frobenius-norm^2 полного Jacobian d(pred)/d(u) (obs_dim=3 каналов, цикл
    по каналам вместо случайной проекции Hoffman et al. — дёшево при
    output_dim=3). create_graph=True обязателен: штраф должен
    backprop'аться в параметры модели через loss.backward() ниже."""
    penalty = torch.zeros((), device=u.device)
    for c in range(pred.shape[-1]):
        grad_c = torch.autograd.grad(
            outputs=pred[..., c], inputs=u,
            grad_outputs=torch.ones_like(pred[..., c]),
            create_graph=True, retain_graph=True,
        )[0]
        penalty = penalty + (grad_c ** 2).sum(dim=-1).mean()
    return penalty


def _train_nrc_strong_jacreg(hidden_dim, batch, device, epochs, lr, gradient_lambda):
    """Конфиг идентичен _train_nrc_strong (5 слоёв, bias=False) — меняется
    ТОЛЬКО регуляризация (Jacobian penalty вместо отсутствия/weight decay)."""
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
    u = u.detach().requires_grad_(True)
    true = batch["true"].to(device)
    loss_fn = NRCRelativeL2Loss(use_confidence_weight=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loss_last, jac_loss_last = None, None
    for ep in range(epochs):
        optimizer.zero_grad()
        pred = model(u)
        data_loss = loss_fn(pred, true)
        if gradient_lambda > 0:
            jac_penalty = _jacobian_penalty(pred, u)
        else:
            jac_penalty = torch.zeros((), device=device)
        loss = data_loss + gradient_lambda * jac_penalty
        loss.backward()
        optimizer.step()
        if ep == epochs - 1:
            data_loss_last = float(data_loss.item())
            jac_loss_last = float(jac_penalty.item())
    model._last_data_loss = data_loss_last
    model._last_jac_loss = jac_loss_last
    return model


def _calibrate_gradient_lambda(seqs, hidden_dim, epochs, lr):
    """Подбор gradient_lambda ДО основного сравнения, по val-loss на
    калибровочных сидах — НЕ по dark-frame MSE, НЕ по сравнению с nrc_kan
    (симметрично _calibrate_weight_decay в run_hypothesis1_reg_nrc_strong.py)."""
    loss_fn = NRCRelativeL2Loss(use_confidence_weight=False)
    mean_val_loss = {gl: [] for gl in GRADIENT_LAMBDA_GRID}
    for seed in range(GRAD_CALIB_SEEDS):
        rng = np.random.default_rng(seed)
        train_idx = tile_stratified_sample(seqs, N_TRAIN_PIXELS, rng)
        full_batch = build_batch_dense(seqs, train_idx)
        n_val = max(1, int(round(N_TRAIN_PIXELS * GRAD_VAL_FRACTION)))
        perm = rng.permutation(N_TRAIN_PIXELS)
        val_pos, sub_pos = perm[:n_val], perm[n_val:]
        sub_batch = {k: (v[sub_pos] if torch.is_tensor(v) else v)
                     for k, v in full_batch.items()}
        val_batch = {k: (v[val_pos] if torch.is_tensor(v) else v)
                     for k, v in full_batch.items()}
        for gl in GRADIENT_LAMBDA_GRID:
            torch.manual_seed(seed)
            m = _train_nrc_strong_jacreg(hidden_dim, sub_batch, NRC_DEVICE, epochs, lr, gl)
            with torch.no_grad():
                val_loss = float(loss_fn(
                    torch.as_tensor(predict_nrc_strong(m, val_batch, NRC_DEVICE)),
                    val_batch["true"],
                ).item())
            mean_val_loss[gl].append(val_loss)

    avg = {gl: float(np.mean(v)) for gl, v in mean_val_loss.items()}
    print("=== калибровка gradient_lambda (val-loss, среднее по "
          f"{GRAD_CALIB_SEEDS} сидам, hidden_dim={hidden_dim}) ===")
    for gl in GRADIENT_LAMBDA_GRID:
        print(f"  gradient_lambda={gl:.0e}: val_loss={avg[gl]:.4e}")
    best_gl = min(avg, key=avg.get)
    print(f"  -> выбрано gradient_lambda={best_gl:.0e}\n")
    return best_gl


def run_hypothesis1b(dataset_path=DEFAULT_DATASET, hidden_dim=HIDDEN_DIM,
                      n_seeds=N_SEEDS, n_train_pixels=N_TRAIN_PIXELS,
                      epochs=EPOCHS, lr=LR, gradient_lambda=None):
    with open(KAN_RESULTS_PATH) as f:
        kan_results = json.load(f)
    with open(H1_RESULTS_PATH) as f:
        h1_results = json.load(f)
    kan_dark = np.array(kan_results["dark_by"]["nrc_kan"][str(hidden_dim)])
    plain_strong_dark = np.array(kan_results["dark_by"]["nrc_strong"][str(hidden_dim)])
    wd_reg_dark = np.array(h1_results["reg_dark"])
    for name, arr in [("nrc_kan", kan_dark), ("nrc_strong_plain", plain_strong_dark),
                       ("nrc_strong_reg(weight_decay)", wd_reg_dark)]:
        if len(arr) != n_seeds:
            raise ValueError(f"{name}: {len(arr)} сидов, ожидалось {n_seeds}")

    seqs = load_fully_valid_pixel_sequences_zeroday(dataset_path)

    if gradient_lambda is None:
        gradient_lambda = _calibrate_gradient_lambda(seqs, hidden_dim, epochs, lr)

    print(f"=== Гипотеза 1b: nrc_strong_jacreg (gradient_lambda={gradient_lambda:.0e}) "
          f"vs nrc_kan, dark-frame MSE, hidden_dim={hidden_dim}, n_seeds={n_seeds} ===")
    jacreg_dark = []
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

        m = _train_nrc_strong_jacreg(hidden_dim, train_batch, NRC_DEVICE, epochs, lr, gradient_lambda)
        pred = predict_nrc_strong(m, eval_batch, NRC_DEVICE)
        dk, _br = slice_mse_by_brightness(pred, true, n_eval, dark_threshold)
        jacreg_dark.append(dk)
        print(f"  seed {seed+1}/{n_seeds}: nrc_strong_jacreg dark={dk:.3e} "
              f"(nrc_kan[seed]={kan_dark[seed]:.3e}) "
              f"[data_loss={m._last_data_loss:.3e} jac_loss={m._last_jac_loss:.3e}]")

    jacreg_dark = np.array(jacreg_dark)
    sig_vs_kan = paired_bootstrap_significance(kan_dark, jacreg_dark)
    rob_vs_kan = seed_robustness_report(kan_dark, jacreg_dark)
    sig_vs_wd = paired_bootstrap_significance(jacreg_dark, wd_reg_dark)
    rob_vs_wd = seed_robustness_report(jacreg_dark, wd_reg_dark)

    print("\n=== результат ===")
    print(f"nrc_kan vs nrc_strong_jacreg (dark): mean_diff(kan-jacreg)={sig_vs_kan['mean_diff']:.3e} "
          f"CI=[{sig_vs_kan['ci_low']:.3e},{sig_vs_kan['ci_high']:.3e}] "
          f"significant={sig_vs_kan['significant']} | "
          f"cohens_d={rob_vs_kan['cohens_d']:.2f} -> {rob_vs_kan['recommendation']}")
    print(f"nrc_strong_jacreg vs nrc_strong_reg(weight_decay) (dark, сравнение "
          f"двух видов регуляризации): mean_diff(jacreg-wd)={sig_vs_wd['mean_diff']:.3e} "
          f"CI=[{sig_vs_wd['ci_low']:.3e},{sig_vs_wd['ci_high']:.3e}] "
          f"significant={sig_vs_wd['significant']} | cohens_d={rob_vs_wd['cohens_d']:.2f}")

    gap_closed = (not sig_vs_kan["significant"]) or abs(rob_vs_kan["cohens_d"]) < 0.5
    print("\n=== вердикт Гипотезы 1b ===")
    if gap_closed:
        print("  Разрыв KAN vs MLP на тёмных кадрах схлопнулся после Jacobian-"
              "регуляризации nrc_strong (significant=False или |d|<0.5) — эффект "
              "объясняется отсутствием СПЕЦИФИЧЕСКИ гладкостной (не любой) "
              "регуляризации, не архитектурой KAN как таковой. Гипотеза 1 в целом "
              "ПОДТВЕРЖДЕНА при правильно подобранном виде регуляризации.")
    else:
        print("  Разрыв остался значим и устойчив даже после Jacobian-регуляризации "
              "nrc_strong, соответствующей той же механике, что curvature_lambda у "
              "KAN — Гипотеза 1 ОТКЛОНЕНА при обоих проверенных видах регуляризации "
              "(weight decay и gradient/Jacobian penalty). Переходить к Гипотезе 2.")

    results = dict(
        hidden_dim=hidden_dim, n_seeds=n_seeds, gradient_lambda=gradient_lambda,
        gradient_lambda_grid=GRADIENT_LAMBDA_GRID,
        jacreg_dark=jacreg_dark.tolist(), kan_dark=kan_dark.tolist(),
        plain_strong_dark=plain_strong_dark.tolist(), wd_reg_dark=wd_reg_dark.tolist(),
        sig_vs_kan=sig_vs_kan, rob_vs_kan=rob_vs_kan,
        sig_vs_wd=sig_vs_wd, rob_vs_wd=rob_vs_wd,
        gap_closed=bool(gap_closed),
    )
    out_path = Path(dataset_path).parent / "run_hypothesis1b_jacobian_reg_nrc_strong_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                   default=lambda o: bool(o) if isinstance(o, np.bool_) else o)
    print(f"\nСырые результаты сохранены в {out_path}")
    return results
