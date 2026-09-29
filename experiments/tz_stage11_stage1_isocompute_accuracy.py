"""
tz_stage11_stage1_isocompute_accuracy.py — TZ_stage11 §2.3 (26-28.09.2026):
iso-compute (не iso-width) сравнение узкой lmKAN (NRCLmKANBaseline) против
патент-верного nrc_strong — ширина KAN подобрана под вычислительный бюджет
nrc_strong@hidden_dim=64 (w_kan_param = 64/sqrt(1+num_grids) ≈ 19, §2.1).

СЕТКА ШИРИН KAN (решение 28.09.2026, чат): {16, 24, 32} — кратны
tile_size_forward=8 (NRCLmKANBaseline паддит hidden_dim ВВЕРХ до кратного 8),
поэтому номинальная и физическая ширина совпадают (справедливый тест;
исходная ТЗ-сетка {15,19,23} давала физические {16,24,24} — 19 и 23
схлопывались). Decision rule привязан к DECISION_KAN_HIDDEN_DIM=24
(= padded(w_kan_param)).

MLP (nrc_strong): НЕ переобучается — переиспользуются готовые 16-сидовые
результаты TZ_stage8 (reference_results/run_capacity_sweep_zeroday_kan_results.json).

Регуляризация кривизны: hessian_regularization_loss() с decay schedule —
вынесена без изменений в gi_surrogate.training.lmkan вместе с train/predict.

Устройство: ТОЛЬКО CUDA (lmKAN — fused CUDA-кернелы, GPU-only; MPS не имеет
документированных гарантий torch.use_deterministic_algorithms()).

Мониторинг сходимости: NaN/расхождение loss — ОТДЕЛЬНЫЙ ИСХОД ("НЕ ОБУЧАЕТСЯ"),
не "хуже по точности" — см. gi_surrogate.training.lmkan.train_nrc_lmkan.

Decision rule считается ТОЛЬКО в одной паре: KAN@24 vs nrc_strong@64.
Остальные точки сетки — только для кривой точность-от-компьюта (контекст).
"""
import argparse
import json
import math
import resource
import time

import numpy as np
import torch

from gi_surrogate.data.sampling import tile_stratified_sample
from gi_surrogate.data.zeroday_adapter import load_fully_valid_pixel_sequences_zeroday
from gi_surrogate.data.batch import build_batch_dense
from gi_surrogate.eval.slices import slice_mse, slice_mse_by_brightness
from gi_surrogate.eval.stats import paired_bootstrap_significance, seed_robustness_report, MIN_N_SEEDS
from gi_surrogate.models.kan_lmkan import _pad_to_multiple
from gi_surrogate.paths import DEFAULT_DATASET, TZ8_KAN_RESULTS, resolve_out_path
from gi_surrogate.training.common import predict_in_chunks, EVAL_CHUNK_SIZE
from gi_surrogate.training.lmkan import (
    NUM_GRIDS, TILE_SIZE_FORWARD, HESS_LAMBDA_INIT, HESS_LAMBDA_FINAL,
    train_nrc_lmkan, predict_nrc_lmkan,
)
from gi_surrogate.utils.determinism import set_determinism

MLP_REFERENCE_HIDDEN_DIM = 64        # nrc_strong точка, под чей бюджет подбирается w_kan_param (§2.1)
MLP_CONTEXT_HIDDEN_DIMS = [16, 64]   # обе переиспользуются из TZ_stage8 для кривой

W_KAN_PARAM = round(MLP_REFERENCE_HIDDEN_DIM / math.sqrt(1 + NUM_GRIDS))  # ≈19 (теоретическое значение)
DECISION_KAN_HIDDEN_DIM = _pad_to_multiple(W_KAN_PARAM, TILE_SIZE_FORWARD)  # 24
KAN_HIDDEN_DIMS = [DECISION_KAN_HIDDEN_DIM - TILE_SIZE_FORWARD, DECISION_KAN_HIDDEN_DIM,
                   DECISION_KAN_HIDDEN_DIM + TILE_SIZE_FORWARD]  # [16, 24, 32]

N_SEEDS = 16   # тот же протокол, что TZ_stage8


def _log_padding_collisions(hidden_dims, tile_size=TILE_SIZE_FORWARD):
    """Явная проверка/лог коллизий padded-ширины, вместо тихого дублирования
    конфигураций (см. докстринг модуля)."""
    padded = {h: _pad_to_multiple(h, tile_size) for h in hidden_dims}
    print(f"padded hidden_dim map: {padded}")
    seen = {}
    for h, p in padded.items():
        seen.setdefault(p, []).append(h)
    for p, hs in seen.items():
        if len(hs) > 1:
            print(f"  WARNING: requested hidden_dim={hs} все паддятся в физическую "
                  f"ширину {p} — это ОДНА и та же архитектура, посчитанная "
                  f"несколько раз под разными номинальными значениями. Использовать "
                  f"значения, кратные {tile_size}.")
    return padded


def _load_mlp_reference(old_results_path, hidden_dims, n_seeds):
    """Готовые (НЕ переобучаемые) nrc_strong-точки из TZ_stage8."""
    with open(old_results_path) as f:
        old = json.load(f)
    ref = {"disocclusion": {}, "stable": {}, "dark": {}, "bright": {}}
    key_map = {"disocclusion": "disocc_by", "stable": "stable_by",
               "dark": "dark_by", "bright": "bright_by"}
    for h in hidden_dims:
        if str(h) not in old["disocc_by"]["nrc_strong"]:
            raise ValueError(f"hidden_dim={h} отсутствует в {old_results_path} "
                              f"(доступно: {old['capacity_levels']})")
        n_have = len(old["disocc_by"]["nrc_strong"][str(h)])
        if n_have != n_seeds:
            raise ValueError(f"{old_results_path} содержит {n_have} сидов для "
                              f"hidden_dim={h}, ожидалось {n_seeds}")
        for slice_name, json_key in key_map.items():
            ref[slice_name][h] = list(old[json_key]["nrc_strong"][str(h)])
    return ref


def run_stage1(dataset_path=DEFAULT_DATASET, n_seeds=N_SEEDS, n_train_pixels=90,
                epochs=200, lr=1e-3, kan_hidden_dims=None,
                hess_lambda_init=HESS_LAMBDA_INIT, hess_lambda_final=HESS_LAMBDA_FINAL,
                eval_chunk_size=EVAL_CHUNK_SIZE, out_dir=None):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TZ_stage11 §4: Stage 1 требует CUDA (lmKAN — fused CUDA-кернелы, "
            "GPU-only; отдельно, MPS не имеет документированных гарантий "
            "torch.use_deterministic_algorithms()) — запускать на Kaggle T4x2, "
            "не локально на M1/MPS."
        )
    device = torch.device("cuda")
    kan_hidden_dims = kan_hidden_dims or KAN_HIDDEN_DIMS
    if n_seeds < MIN_N_SEEDS:
        raise ValueError(f"n_seeds={n_seeds} < {MIN_N_SEEDS} (gi_surrogate.eval.stats.MIN_N_SEEDS)")

    padded_map = _log_padding_collisions(kan_hidden_dims)
    if DECISION_KAN_HIDDEN_DIM not in kan_hidden_dims:
        raise ValueError(f"kan_hidden_dims={kan_hidden_dims} не содержит decision-точку "
                          f"DECISION_KAN_HIDDEN_DIM={DECISION_KAN_HIDDEN_DIM} — decision rule "
                          f"не может быть вычислен.")
    mlp_ref = _load_mlp_reference(TZ8_KAN_RESULTS, MLP_CONTEXT_HIDDEN_DIMS, n_seeds)

    seqs = load_fully_valid_pixel_sequences_zeroday(dataset_path)
    print(f"device: {device}, w_kan_param={W_KAN_PARAM} (§2.1), "
          f"kan_hidden_dims={kan_hidden_dims}, n_seeds={n_seeds}, "
          f"hess_lambda: {hess_lambda_init:.1e} -> {hess_lambda_final:.1e}, "
          f"epochs={epochs}, n_train_pixels={n_train_pixels}, "
          f"n_pixels_total={len(seqs)}", flush=True)

    disocc_by = {h: [] for h in kan_hidden_dims}
    stable_by = {h: [] for h in kan_hidden_dims}
    dark_by = {h: [] for h in kan_hidden_dims}
    bright_by = {h: [] for h in kan_hidden_dims}
    diverged_seeds = {h: [] for h in kan_hidden_dims}
    valid_seeds = {h: [] for h in kan_hidden_dims}
    n_params = {}
    train_time_s = {h: [] for h in kan_hidden_dims}

    for seed in range(n_seeds):
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        print(f"\n--- seed {seed+1}/{n_seeds} --- (peak RSS: {peak_rss_mb:.0f} MB)", flush=True)
        set_determinism(seed, warn_only=True)
        rng = np.random.default_rng(seed)
        train_idx = tile_stratified_sample(seqs, n_train_pixels, rng)
        train_idx_set = set(train_idx)
        eval_idx = [i for i in range(len(seqs)) if i not in train_idx_set]
        train_batch = build_batch_dense(seqs, train_idx)
        eval_batch = build_batch_dense(seqs, eval_idx)
        true = eval_batch["true"].numpy()
        disocc_flags_arr = eval_batch["disocclusion_flag"].numpy()
        n_eval = len(eval_idx)
        dark_threshold = float(np.median(true))  # per-seed, честный порог (тот же принцип, что TZ_stage8)

        for hdim in kan_hidden_dims:
            t0 = time.perf_counter()
            print(f"  [seed {seed+1}] nrc_lmkan hidden_dim={hdim} "
                  f"(padded={padded_map[hdim]})...", end="", flush=True)
            model, diverged = train_nrc_lmkan(
                hdim, train_batch, device, epochs, lr, hess_lambda_init, hess_lambda_final)
            dt = time.perf_counter() - t0
            if hdim not in n_params:
                n_params[hdim] = sum(p.numel() for p in model.parameters())
            if diverged:
                diverged_seeds[hdim].append(seed)
                print(f" DIVERGED ({dt:.1f}s) — исключён из массивов метрик", flush=True)
                continue
            pred = predict_in_chunks(predict_nrc_lmkan, model, eval_batch, device, eval_chunk_size)
            d, s = slice_mse(pred, true, disocc_flags_arr, n_eval)
            dk, br = slice_mse_by_brightness(pred, true, n_eval, dark_threshold)
            disocc_by[hdim].append(d)
            stable_by[hdim].append(s)
            dark_by[hdim].append(dk)
            bright_by[hdim].append(br)
            valid_seeds[hdim].append(seed)
            train_time_s[hdim].append(dt)
            print(f" disocc={d:.3e} stable={s:.3e} dark={dk:.3e} bright={br:.3e} ({dt:.1f}s) "
                  f"[data_loss={model._last_data_loss:.3e} hess_loss={model._last_hess_loss:.3e}]",
                  flush=True)

    all_by_kan = {"disocclusion": disocc_by, "stable": stable_by, "dark": dark_by, "bright": bright_by}
    print("\n=== summary (mean MSE_raw over VALID seeds only) — контекстная таблица, не decision rule ===")
    for slice_name in ("disocclusion", "stable", "dark", "bright"):
        print(f"\n-- {slice_name} --")
        for hdim in kan_hidden_dims:
            vals = all_by_kan[slice_name][hdim]
            mean_v = np.mean(vals) if vals else float("nan")
            print(f"  nrc_lmkan  hidden_dim={hdim:<3} (padded={padded_map[hdim]:<3}): "
                  f"{mean_v:.3e}  (valid_seeds={len(vals)}/{n_seeds}, diverged={len(diverged_seeds[hdim])})")
        for hdim in MLP_CONTEXT_HIDDEN_DIMS:
            mean_v = np.mean(mlp_ref[slice_name][hdim])
            print(f"  nrc_strong hidden_dim={hdim:<3} (реюз TZ_stage8):      "
                  f"{mean_v:.3e}  (n_seeds={len(mlp_ref[slice_name][hdim])})")

    print("\n=== мониторинг сходимости (отдельная категория, не 'хуже по точности') ===")
    not_trainable = []
    for hdim in kan_hidden_dims:
        n_valid = len(valid_seeds[hdim])
        print(f"  hidden_dim={hdim}: valid={n_valid}/{n_seeds}, diverged_seeds={diverged_seeds[hdim]}")
        if n_valid < MIN_N_SEEDS:
            not_trainable.append(hdim)

    print(f"\n=== decision rule: KAN@hidden_dim={DECISION_KAN_HIDDEN_DIM} (padded w_kan_param={W_KAN_PARAM}) "
          f"vs nrc_strong@{MLP_REFERENCE_HIDDEN_DIM} ===")
    decision_significance, decision_robustness = {}, {}
    if DECISION_KAN_HIDDEN_DIM in not_trainable:
        print(f"  ИСХОД: НЕ ОБУЧАЕТСЯ — valid_seeds={len(valid_seeds[DECISION_KAN_HIDDEN_DIM])} < "
              f"{MIN_N_SEEDS} при hidden_dim={DECISION_KAN_HIDDEN_DIM} даже с регуляризацией кривизны "
              f"(hess_lambda {hess_lambda_init:.1e}->{hess_lambda_final:.1e}). Направление "
              f"закрывается по практической применимости, НЕ по сравнению точности.")
        verdict = "NOT_TRAINABLE"
    else:
        worse_slices, better_slices = [], []
        for slice_name in ("disocclusion", "stable", "dark", "bright"):
            a = np.array([all_by_kan[slice_name][DECISION_KAN_HIDDEN_DIM][valid_seeds[DECISION_KAN_HIDDEN_DIM].index(s)]
                          for s in valid_seeds[DECISION_KAN_HIDDEN_DIM]])
            b_full = mlp_ref[slice_name][MLP_REFERENCE_HIDDEN_DIM]
            b = np.array([b_full[s] for s in valid_seeds[DECISION_KAN_HIDDEN_DIM]])
            sig = paired_bootstrap_significance(a, b)
            rob = seed_robustness_report(a, b)
            decision_significance[slice_name] = sig
            decision_robustness[slice_name] = rob
            print(f"  {slice_name}: mean_diff(kan-strong)={sig['mean_diff']:.3e} "
                  f"CI=[{sig['ci_low']:.3e},{sig['ci_high']:.3e}] significant={sig['significant']} "
                  f"| cohens_d={rob['cohens_d']:.2f} -> {rob['recommendation']} "
                  f"| std_kan={a.std(ddof=1):.3e} std_mlp={b.std(ddof=1):.3e}")
            if sig["significant"] and sig["mean_diff"] > 0:
                worse_slices.append(slice_name)
            if sig["significant"] and sig["mean_diff"] < 0:
                better_slices.append(slice_name)

        print("\n=== вердикт ===")
        if worse_slices:
            print(f"  ОПРОВЕРГНУТО: KAN значимо хуже nrc_strong на {worse_slices}.")
            verdict = "REFUTED"
        elif better_slices:
            print(f"  ПОДТВЕРЖДЕНО: KAN не хуже nrc_strong ни на одном из 4 срезов и "
                  f"значимо лучше на {better_slices}.")
            verdict = "CONFIRMED"
        else:
            print(f"  ЧАСТИЧНО ПОДТВЕРЖДЕНО: KAN не хуже ни на одном срезе, но и не "
                  f"значимо лучше ни на одном.")
            verdict = "PARTIALLY_CONFIRMED"

    results = dict(
        w_kan_param=W_KAN_PARAM, decision_kan_hidden_dim=DECISION_KAN_HIDDEN_DIM,
        kan_hidden_dims=kan_hidden_dims, padded_hidden_dim_map=padded_map,
        mlp_context_hidden_dims=MLP_CONTEXT_HIDDEN_DIMS,
        mlp_reference_hidden_dim=MLP_REFERENCE_HIDDEN_DIM,
        disocc_by=disocc_by, stable_by=stable_by, dark_by=dark_by, bright_by=bright_by,
        mlp_reference=mlp_ref,
        diverged_seeds=diverged_seeds, valid_seeds=valid_seeds, not_trainable=not_trainable,
        decision_significance=decision_significance, decision_robustness=decision_robustness,
        verdict=verdict, n_params=n_params, train_time_s=train_time_s,
        n_seeds=n_seeds, n_train_pixels=n_train_pixels, epochs=epochs, lr=lr,
        hess_lambda_init=hess_lambda_init, hess_lambda_final=hess_lambda_final,
        num_grids=NUM_GRIDS,
    )
    out_path = resolve_out_path("tz_stage11_stage1_isocompute_accuracy_results.json", out_dir)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                   default=lambda o: bool(o) if isinstance(o, np.bool_) else o)
    print(f"\nСырые результаты сохранены в {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--n_train_pixels", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kan_hidden_dims", type=int, nargs="+", default=KAN_HIDDEN_DIMS,
                         help="По умолчанию {16,24,32} — кратны 8 (tile_size_forward), "
                              "иначе номинальная и физическая ширина разойдутся.")
    parser.add_argument("--hess_lambda_init", type=float, default=HESS_LAMBDA_INIT)
    parser.add_argument("--hess_lambda_final", type=float, default=HESS_LAMBDA_FINAL)
    parser.add_argument("--eval_chunk_size", type=int, default=EVAL_CHUNK_SIZE)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    run_stage1(
        dataset_path=args.dataset, n_seeds=args.n_seeds,
        n_train_pixels=args.n_train_pixels, epochs=args.epochs, lr=args.lr,
        kan_hidden_dims=args.kan_hidden_dims,
        hess_lambda_init=args.hess_lambda_init, hess_lambda_final=args.hess_lambda_final,
        eval_chunk_size=args.eval_chunk_size, out_dir=args.out_dir,
    )
