"""Тесты NRC-модели со spatial-входом (NRCStyleBaseline).

Перенесены из tests/test_spatial_features.py, оставлен только NRC-случай
параметризации (CfC/GRU/ODE-LSTM остались в архиве). Логика проверок не менялась.
"""
import torch

from gi_surrogate.models.nrc import NRCStyleBaseline
from gi_surrogate.models.spatial_features import build_spatial_conditioning, spatial_dim

EXTRA_KWARGS = dict(use_staleness=True, staleness_dim=4)


def _toy_vecs(B=3, T=7):
    position = torch.randn(B, T, 3)
    direction = torch.randn(B, T, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    normal = torch.randn(B, T, 3)
    albedo = torch.rand(B, T, 3)
    return position, direction, normal, albedo


def test_build_input_backward_compat_spatial_none():
    """spatial=None (значение по умолчанию) даёт БИТ-В-БИТ то же поведение, что и до
    добавления параметра; spatial_dim=0 по умолчанию -> input_dim не расширяется."""
    B, T, obs_dim = 2, 6, 1
    obs = torch.randn(B, T, obs_dim)
    cold = torch.zeros(B, T)
    conf = torch.rand(B, T)
    disocc = torch.zeros(B, T)
    mismatch = torch.zeros(B, T)

    u_no_spatial_kw = NRCStyleBaseline.build_input(
        obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch],
    )
    u_explicit_none = NRCStyleBaseline.build_input(
        obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch], spatial=None,
    )
    assert torch.equal(u_no_spatial_kw, u_explicit_none)
    NRCStyleBaseline(obs_dim=obs_dim, hidden_dim=8, **EXTRA_KWARGS)
    assert u_no_spatial_kw.shape[-1] == obs_dim + 4  # staleness_dim=4


def test_spatial_dim_extends_input_and_gives_gradients():
    """NRC принимает spatial_dim>0, расширяет input_dim ровно на spatial_dim и даёт
    реальные градиенты через spatial (backward доходит до параметров модели)."""
    B, T, obs_dim, SD = 3, 8, 1, spatial_dim()
    obs = torch.randn(B, T, obs_dim)
    cold = torch.zeros(B, T)
    conf = torch.rand(B, T)
    disocc = torch.zeros(B, T)
    mismatch = torch.zeros(B, T)
    position, direction, normal, albedo = _toy_vecs(B, T)
    spatial = build_spatial_conditioning(position, direction, normal, albedo)
    assert spatial.shape[-1] == SD

    model = NRCStyleBaseline(obs_dim=obs_dim, hidden_dim=8, spatial_dim=SD, **EXTRA_KWARGS)
    u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                      extra_staleness=[disocc, mismatch], spatial=spatial)
    assert u.shape[-1] == obs_dim + 4 + SD

    pred = model(u)
    assert pred.shape == obs.shape
    pred.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
