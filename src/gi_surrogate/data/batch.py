"""
dense_batch_core.py — TZ_stage11 (26.09.2026, чат): облегчённая копия
минимально необходимого подмножества run_experiment.py::build_batch/
_stack_feature, run_experiment_dense.py::build_batch_dense/get_spatial/
SPATIAL_DIM и run_hybrid_v0.py::_slice_mse.

ПОЧЕМУ ДУБЛИКАТ, А НЕ ИМПОРТ: run_experiment_dense.py и run_hybrid_v0.py на
уровне модуля безусловно импортируют models/temporal/cfc_probe_module.py
(CfC-архитектура, TZ_stage4-7 — направление закрыто 2026-09-12, TZ_stage7
§9, см. /areas/lnn-lowlight.md) и run_experiment.py/run_experiment_blender.py
целиком — весь этот легаси-хвост проекту для теста KAN vs nrc_strong не
нужен. Решение 26.09.2026 (чат): GitHub должен содержать только файлы,
реально нужные для текущего Kaggle-теста, без CfC — отсюда этот файл, а не
рефакторинг run_experiment_dense.py/run_hybrid_v0.py (те остаются
НЕТРОНУТЫМИ, изменение общей продакшн-инфраструктуры ради одного скрипта —
не сделано, риск не оправдан).

Логика функций ниже — ТОЧНАЯ копия оригиналов на момент 26.09.2026 (не
переписана, не "улучшена" по пути) — при расхождении с оригиналами в
будущем ориентир для истины — run_experiment.py/run_experiment_dense.py/
run_hybrid_v0.py (эти три остаются каноническим источником вне зависимости
от того, что попадает на GitHub).
"""
import numpy as np
import torch

from gi_surrogate.models.spatial_features import build_spatial_conditioning, spatial_dim as _spatial_dim_fn

# Идентично run_experiment_dense.py — единая размерность spatial-входа для
# ВСЕХ "_spatial"-конфигураций (roughness не передаётся, см. models/
# spatial_features.py докстринг — в Zero Day это константа материала).
SPATIAL_DIM = _spatial_dim_fn(n_freqs_position=4, n_freqs_direction=2, include_roughness=False)

# Идентично run_experiment_dense.py — половина протяжённости пола сцены
# (bpy.ops.mesh.primitive_plane_add(size=8) -> -4..+4), нормализация position
# перед positional encoding (2026-08-20 багфикс).
FLOOR_HALF_EXTENT = 4.0


def _stack_feature(seqs, indices, key, T):
    """Идентично run_experiment.py::_stack_feature."""
    arr = np.stack([seqs[i][key][:T] for i in indices])
    if arr.ndim == 2:  # [n_probes, T] -> [n_probes, T, 1]
        arr = arr[..., None]
    return torch.tensor(arr, dtype=torch.float32)


def build_batch(seqs, indices):
    """Идентично run_experiment.py::build_batch."""
    T = min(len(seqs[i]["t"]) for i in indices)
    obs = _stack_feature(seqs, indices, "obs", T)
    dt = torch.tensor(np.stack([seqs[i]["dt"][:T] for i in indices]), dtype=torch.float32)
    cold = torch.tensor(np.stack([seqs[i]["cold_start"][:T] for i in indices]), dtype=torch.float32)
    spp = torch.tensor(np.stack([seqs[i]["spp"][:T] for i in indices]), dtype=torch.float32)
    conf = torch.log1p(spp) / np.log1p(64.0)
    true = _stack_feature(seqs, indices, "true_irradiance", T)
    t_arr = np.stack([seqs[i]["t"][:T] for i in indices])
    return dict(obs=obs, dt=dt, cold=cold, conf=conf, true=true, t=t_arr)


def build_batch_dense(seqs, indices):
    """Идентично run_experiment_dense.py::build_batch_dense (там —
    build_batch(seqs, indices) из run_experiment.py + dense-специфичные
    каналы; здесь — build_batch() выше, та же функция, без изменений)."""
    batch = build_batch(seqs, indices)
    T = batch["dt"].shape[1]
    batch["disocclusion_flag"] = torch.tensor(
        np.stack([seqs[i]["disocclusion_flag"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["geometric_mismatch_score"] = torch.tensor(
        np.stack([seqs[i]["geometric_mismatch_score"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["valid"] = torch.tensor(
        np.stack([seqs[i]["valid"][:T] for i in indices]), dtype=torch.bool,
    )
    batch["position"] = torch.tensor(
        np.stack([seqs[i]["position"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["normal"] = torch.tensor(
        np.stack([seqs[i]["normal"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["albedo"] = torch.tensor(
        np.stack([seqs[i]["albedo"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["direction"] = torch.tensor(
        np.stack([seqs[i]["direction"][:T] for i in indices]), dtype=torch.float32,
    )
    return batch


def get_spatial(batch, device):
    """Идентично run_experiment_dense.py::get_spatial."""
    return build_spatial_conditioning(
        batch["position"].to(device), batch["direction"].to(device),
        batch["normal"].to(device), batch["albedo"].to(device),
        position_scale=FLOOR_HALF_EXTENT,
    )
