"""Замер online-скорости (forward+backward на потоке батчей) — общий для TZ_stage9 и TZ_stage11 Stage 0.

Вынесено без изменений логики из run_tz_stage9_online_speed.py.
"""
import numpy as np
import time
import torch
from gi_surrogate.data.batch import build_batch_dense, get_spatial

STRONG_S_BATCHES = 4         # Müller et al. 2021, arXiv 2106.12372, "Amortization..."

STRONG_L_RECORDS = 16384     # Müller et al. 2021, arXiv 2106.12372, "Amortization..."


def build_step_batch(seqs, idx, device):
    """Собирает вход/таргет для ОДНОГО online-шага (малый батч пикселей,
    §3.3 ТЗ) — тот же build_batch_dense, что offline-протокол TZ_stage6-9,
    но вызывается заново на КАЖДЫЙ шаг с новым случайным idx, а не один раз
    на train/eval сплит."""
    batch = build_batch_dense(seqs, idx)
    obs = batch["obs"].to(device)
    cold, conf = batch["cold"].to(device), batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    true = batch["true"].to(device)
    spatial = get_spatial(batch, device)
    return obs, cold, conf, disocc, mismatch, true, spatial


def timed_online_run(model, build_input_fn, seqs, obs_dim, batch_size, n_frames,
                       s_batches, warmup_frames, device, rng):
    """§3.3-3.4 ТЗ (числа протокола — Müller et al. 2021, см. докстринг
    модуля): n_frames "кадров", каждый — s_batches последовательных
    forward+backward шагов на СЛУЧАЙНОМ батче из batch_size пикселей (не
    train/eval сплит — online-профиль). warmup_frames прогоняются и
    отбрасываются (CUDA JIT/аллокатор warm-up — честность замера, которой
    сознательно не было в Шаге 0 §3.0, там это была ГРУБАЯ прикидка, здесь
    — протокольный замер).
    Возвращает: per-step forward-only [N], per-step forward+backward [N],
    per-frame forward+backward (сумма s_batches подряд идущих шагов) [n_frames].
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    fwd_times, fwdbwd_times = [], []
    frame_totals = []

    for frame in range(warmup_frames + n_frames):
        frame_total = 0.0
        for _ in range(s_batches):
            idx = rng.choice(len(seqs), size=min(batch_size, len(seqs)), replace=False)
            obs, cold, conf, disocc, mismatch, true, spatial = build_step_batch(seqs, idx, device)
            u = build_input_fn(obs, cold, conf, use_staleness=True,
                                extra_staleness=[disocc, mismatch], spatial=spatial)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(u)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            loss = loss_fn(pred, true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            if frame >= warmup_frames:
                fwd_times.append((t1 - t0) * 1000)
                fwdbwd_times.append((t2 - t0) * 1000)
                frame_total += (t2 - t0) * 1000
        if frame >= warmup_frames:
            frame_totals.append(frame_total)

    return np.array(fwd_times), np.array(fwdbwd_times), np.array(frame_totals)
