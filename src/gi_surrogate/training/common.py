"""Общие вспомогательные функции обучения/инференса."""
import numpy as np
import torch

# ИНЦИДЕНТ 14.09.2026 (после первого запуска на машине с 8GB RAM): без
# батчинга predict_nrc_kan прогоняет весь eval (~57.5k пикселей x 61 кадр
# ≈ 3.5М точек) ОДНИМ forward pass — KANLinear.b_splines() первого слоя
# (in_features=55) создаёт тензор [N, 55, grid_size+spline_order=8] ≈ 6.2GB
# на ОДНОМ промежуточном тензоре ОДНОГО слоя, что почти полностью съедает
# 8GB RAM машины. Итог: swap-thrashing (800.6с на seed 1/hidden_dim=16
# вместо ожидаемых по чистому времени вычисления ~35с), затем процесс убит
# ОС на hidden_dim=64 (лог обрывается без Python-трейсбека — сигнатура OOM-
# kill, не управляемое исключение). EVAL_CHUNK_SIZE=2000 держит этот тензор
# в пределах ~210MB на чанк (2000*61*55*8*4 байт) независимо от общего
# размера eval-множества — safety margin, не тонкая настройка под скорость.
EVAL_CHUNK_SIZE = 2000


def predict_in_chunks(predict_fn, model, batch, device, chunk_size=EVAL_CHUNK_SIZE):
    """Батчит предсказание по оси пикселей (dim 0 у всех тензоров
    build_batch_dense) — см. докстринг EVAL_CHUNK_SIZE выше, почему это
    обязательно для nrc_kan на полном eval-множестве. Для nrc_strong
    memory footprint на порядки меньше (обычный MLP, без b-spline
    тензоров), но чанкается тем же путём для единообразия/устойчивости —
    накладные расходы на чанкинг пренебрежимо малы относительно самого
    forward pass."""
    n = batch["obs"].shape[0]
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sub_batch = {k: (v[start:end] if torch.is_tensor(v) else v) for k, v in batch.items()}
        chunks.append(predict_fn(model, sub_batch, device))
    return np.concatenate(chunks, axis=0)
