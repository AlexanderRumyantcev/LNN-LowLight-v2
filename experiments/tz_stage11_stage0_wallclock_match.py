"""
tz_stage11_stage0_wallclock_match.py — TZ_stage11 §2.2 (25.09.2026).

Находит w_kan_wallclock: ширину KAN (lmKAN, тот же harness, что TZ_stage9
§3.3-3.4), при которой online forward+backward время ближе всего к
nrc_strong@hidden_dim=64 (референсная точка, TZ_stage9: 64.563мс/шаг).

Не переобучает nrc_strong — использует готовое число 64.563мс (TZ_stage9,
Kaggle T4x2) как референс, чтобы не тратить GPU-время на то, что уже
измерено на той же платформе тем же протоколом.
"""
import argparse
import json

import numpy as np
import torch

from gi_surrogate.data.zeroday_adapter import load_fully_valid_pixel_sequences_zeroday
from gi_surrogate.data.batch import SPATIAL_DIM
from gi_surrogate.models.kan_lmkan import NRCLmKANBaseline
from gi_surrogate.paths import DEFAULT_DATASET, resolve_out_path
from gi_surrogate.training.nrc import STRONG_N_HIDDEN_LAYERS
from gi_surrogate.training.timing import STRONG_S_BATCHES, STRONG_L_RECORDS, timed_online_run

NRC_STRONG_HD64_MS = 64.563  # TZ_stage9 §3.3-3.4, Kaggle T4x2, эмпирика — не пересчитывается
KAN_WIDTH_GRID = [6, 8, 10, 12, 16]  # смещено к экстраполированной оценке w≈7.9, не к iso-param≈19


def run(dataset_path=DEFAULT_DATASET, batch_size=STRONG_L_RECORDS, n_frames=25,
        s_batches=STRONG_S_BATCHES, warmup_frames=5, width_grid=KAN_WIDTH_GRID, seed=0,
        out_dir=None):
    assert torch.cuda.is_available(), "требует CUDA (fused-kernel lmKAN)"
    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    seqs = load_fully_valid_pixel_sequences_zeroday(dataset_path)
    obs_dim = 3
    print(f"device={device}, референс nrc_strong@64={NRC_STRONG_HD64_MS}мс "
          f"(TZ_stage9, не переобучается), width_grid={width_grid}", flush=True)

    results = {}
    for w in width_grid:
        kan = NRCLmKANBaseline(
            obs_dim=obs_dim, hidden_dim=w, use_staleness=True, staleness_dim=4,
            spatial_dim=SPATIAL_DIM, n_hidden_layers=STRONG_N_HIDDEN_LAYERS,
        ).to(device)
        _, fwdbwd_kan, _ = timed_online_run(
            kan, NRCLmKANBaseline.build_input, seqs, obs_dim, batch_size, n_frames,
            s_batches, warmup_frames, device, rng)
        mean_ms = fwdbwd_kan.mean()
        diff_pct = (mean_ms / NRC_STRONG_HD64_MS - 1.0) * 100
        print(f"  hidden_dim={w:>3}: KAN шаг(fwd+bwd)={mean_ms:.3f}мс "
              f"(p50={np.median(fwdbwd_kan):.3f}) — {diff_pct:+.1f}% к nrc_strong@64", flush=True)
        results[w] = {"kan_fwdbwd_ms": fwdbwd_kan.tolist(), "mean_ms": float(mean_ms),
                       "diff_pct_vs_nrc_strong64": float(diff_pct)}

    w_best = min(results, key=lambda w: abs(results[w]["diff_pct_vs_nrc_strong64"]))
    print(f"\nw_kan_wallclock = {w_best} "
          f"(ближайшее к nrc_strong@64={NRC_STRONG_HD64_MS}мс, "
          f"{results[w_best]['diff_pct_vs_nrc_strong64']:+.1f}%)")

    out_path = resolve_out_path("tz_stage11_stage0_results.json", out_dir)
    with open(out_path, "w") as f:
        json.dump({"nrc_strong_hd64_ms": NRC_STRONG_HD64_MS, "width_grid": width_grid,
                   "results": results, "w_kan_wallclock": w_best}, f, indent=2)
    print(f"Сохранено: {out_path}")
    return results, w_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--batch_size", type=int, default=STRONG_L_RECORDS)
    parser.add_argument("--n_frames", type=int, default=25)
    parser.add_argument("--s_batches", type=int, default=STRONG_S_BATCHES)
    parser.add_argument("--warmup_frames", type=int, default=5)
    parser.add_argument("--width_grid", type=int, nargs="+", default=KAN_WIDTH_GRID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    run(dataset_path=args.dataset, batch_size=args.batch_size, n_frames=args.n_frames,
        s_batches=args.s_batches, warmup_frames=args.warmup_frames,
        width_grid=args.width_grid, seed=args.seed, out_dir=args.out_dir)
