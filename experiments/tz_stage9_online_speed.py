"""
tz_stage9_online_speed.py — TZ_stage9 §3.3-3.4 (18.09.2026): online-замер
задержки nrc_strong (обычный PyTorch MLP) vs NRCLmKANBaseline (fused-kernel
KAN) на ОДНОЙ и той же GPU (Kaggle T4).

ЗАПУСК: только на Kaggle T4x2, после установки lmkan (git clone
schwallergroup/lmkan && pip install .) и после git push текущих локальных
изменений.

ЧТО ИЗМЕРЯЕТСЯ (§3.3 ТЗ, верифицированный протокол Müller et al. 2021,
arXiv 2106.12372, "Amortization in a Real-time Path Tracer"): s=4
последовательных forward+backward шагов по l=16384 сэмплов каждый —
STRONG_S_BATCHES/STRONG_L_RECORDS (gi_surrogate.training.timing), не
"реалистичное приближение на глаз" (первая версия скрипта до исправления
использовала batch_size=1024 без проверки по первоисточнику — тот же класс
ошибки, что был с CfC, см. EXPERIMENT_LOG.md "ПОПРАВКА 2026-09-12").

ЧТО НЕ ИЗМЕРЯЕТСЯ (§3.4/§5 ТЗ): точность моделей — отдельно, TZ_stage11.

Логика замера (build_step_batch/timed_online_run) вынесена без изменений в
gi_surrogate.training.timing — этот файл только оркестрирует прогон по
capacity_levels и печатает/сохраняет сводку.
"""
import argparse
import json

import numpy as np
import torch

from gi_surrogate.data.zeroday_adapter import load_fully_valid_pixel_sequences_zeroday
from gi_surrogate.data.batch import SPATIAL_DIM
from gi_surrogate.models.nrc import NRCStyleBaseline
from gi_surrogate.models.kan_lmkan import NRCLmKANBaseline
from gi_surrogate.paths import DEFAULT_DATASET, resolve_out_path
from gi_surrogate.training.nrc import STRONG_N_HIDDEN_LAYERS, STRONG_BIAS
from gi_surrogate.training.timing import STRONG_S_BATCHES, STRONG_L_RECORDS, timed_online_run

CAPACITY_LEVELS = [16, 64]   # TZ_stage9 §3.2, переиспользованы точки TZ_stage8


def run(dataset_path=DEFAULT_DATASET, batch_size=STRONG_L_RECORDS, n_frames=25,
        s_batches=STRONG_S_BATCHES, warmup_frames=5, capacity_levels=CAPACITY_LEVELS, seed=0,
        out_dir=None):
    assert torch.cuda.is_available(), "TZ_stage9 требует CUDA (fused-kernel lmKAN) — CPU не подходит"
    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    seqs = load_fully_valid_pixel_sequences_zeroday(dataset_path)
    obs_dim = 3  # проверено напрямую по датасету 18.09.2026
    print(f"device={device}, batch_size={batch_size} (Müller et al. 2021: l=16384), "
          f"s_batches={s_batches} (Müller et al. 2021: s=4), n_frames={n_frames} "
          f"(+{warmup_frames} warmup), n_pixels_total={len(seqs)}, "
          f"capacity_levels={capacity_levels}", flush=True)

    results = {}
    for hdim in capacity_levels:
        print(f"\n--- hidden_dim={hdim} ---", flush=True)

        nrc = NRCStyleBaseline(
            obs_dim=obs_dim, hidden_dim=hdim, use_staleness=True, staleness_dim=4,
            spatial_dim=SPATIAL_DIM, n_hidden_layers=STRONG_N_HIDDEN_LAYERS, bias=STRONG_BIAS,
        ).to(device)
        fwd_nrc, fwdbwd_nrc, frames_nrc = timed_online_run(
            nrc, NRCStyleBaseline.build_input, seqs, obs_dim, batch_size, n_frames,
            s_batches, warmup_frames, device, rng)
        print(f"  nrc_strong:  шаг(fwd+bwd)={fwdbwd_nrc.mean():.3f}мс "
              f"(p50={np.median(fwdbwd_nrc):.3f}, p95={np.percentile(fwdbwd_nrc,95):.3f})  "
              f"КАДР(s={s_batches} шагов)={frames_nrc.mean():.3f}мс "
              f"(p50={np.median(frames_nrc):.3f}, p95={np.percentile(frames_nrc,95):.3f})", flush=True)

        kan = NRCLmKANBaseline(
            obs_dim=obs_dim, hidden_dim=hdim, use_staleness=True, staleness_dim=4,
            spatial_dim=SPATIAL_DIM, n_hidden_layers=STRONG_N_HIDDEN_LAYERS,
        ).to(device)
        fwd_kan, fwdbwd_kan, frames_kan = timed_online_run(
            kan, NRCLmKANBaseline.build_input, seqs, obs_dim, batch_size, n_frames,
            s_batches, warmup_frames, device, rng)
        print(f"  kan(lmkan):  шаг(fwd+bwd)={fwdbwd_kan.mean():.3f}мс "
              f"(p50={np.median(fwdbwd_kan):.3f}, p95={np.percentile(fwdbwd_kan,95):.3f})  "
              f"КАДР(s={s_batches} шагов)={frames_kan.mean():.3f}мс "
              f"(p50={np.median(frames_kan):.3f}, p95={np.percentile(frames_kan,95):.3f})", flush=True)

        ratio_fwdbwd = fwdbwd_kan.mean() / fwdbwd_nrc.mean()
        ratio_frame = frames_kan.mean() / frames_nrc.mean()
        print(f"  RATIO (шаг, kan/nrc_strong): {ratio_fwdbwd:.2f}x   "
              f"RATIO (кадр, kan/nrc_strong): {ratio_frame:.2f}x", flush=True)
        print(f"  для справки: бюджет настоящего NRC на весь кэш ~2.6мс/кадр "
              f"(Müller et al. 2021, fused-kernel — НЕ прямое сравнение, другой "
              f"класс реализации, см. TZ_stage9 §3.4/§5)", flush=True)

        results[hdim] = {
            "nrc_strong": {"fwdbwd_ms": fwdbwd_nrc.tolist(), "frame_ms": frames_nrc.tolist()},
            "kan_lmkan": {"fwdbwd_ms": fwdbwd_kan.tolist(), "frame_ms": frames_kan.tolist()},
            "ratio_fwdbwd": ratio_fwdbwd,
            "ratio_frame": ratio_frame,
        }

    print("\n=== §4 TZ_stage9 decision rule (предварительно, по обеим точкам) ===")
    ratios = [results[h]["ratio_fwdbwd"] for h in capacity_levels]
    max_ratio = max(ratios)
    if max_ratio <= 5.0:
        verdict = "ДА (предварительно) — разрыв закрылся до разумных пределов на обеих точках"
    elif max_ratio >= 10.0:
        verdict = "НЕТ (предварительно) — разрыв остаётся на порядок и больше даже с fused-kernel"
    else:
        verdict = "ПРОМЕЖУТОЧНО — требует решения (PolyKAN как 2й кандидат, либо явный бюджет/кадр)"
    print(f"  max ratio (fwd+bwd, по {capacity_levels}) = {max_ratio:.2f}x -> {verdict}")

    out_path = resolve_out_path("tz_stage9_online_speed_results.json", out_dir)
    with open(out_path, "w") as f:
        json.dump({"batch_size": batch_size, "s_batches": s_batches, "n_frames": n_frames,
                   "results": results, "max_ratio_fwdbwd": max_ratio, "verdict": verdict},
                  f, indent=2)
    print(f"\nСырые результаты сохранены в {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--batch_size", type=int, default=STRONG_L_RECORDS,
                         help="l в терминах Müller et al. 2021 (по умолчанию 16384, как в статье)")
    parser.add_argument("--n_frames", type=int, default=25)
    parser.add_argument("--s_batches", type=int, default=STRONG_S_BATCHES,
                         help="s в терминах Müller et al. 2021 (по умолчанию 4, как в статье)")
    parser.add_argument("--warmup_frames", type=int, default=5)
    parser.add_argument("--capacity_levels", type=int, nargs="+", default=CAPACITY_LEVELS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    run(dataset_path=args.dataset, batch_size=args.batch_size, n_frames=args.n_frames,
        s_batches=args.s_batches, warmup_frames=args.warmup_frames,
        capacity_levels=args.capacity_levels, seed=args.seed, out_dir=args.out_dir)
