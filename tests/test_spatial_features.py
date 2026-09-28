"""Тесты spatial-conditioning (positional encoding, spatial_dim, нормировка position).

Перенесены из tests/test_spatial_features.py; тесты, завязанные на CfC/GRU/ODE-LSTM
(параметризованные проверки build_input и влияния spatial-входа) остались в архиве.
Проверки NRC-модели с spatial-входом - в tests/test_nrc_spatial.py.
"""
import pytest
import torch

from gi_surrogate.models.spatial_features import build_spatial_conditioning, positional_encoding, spatial_dim


def _toy_vecs(B=3, T=7):
    position = torch.randn(B, T, 3)
    direction = torch.randn(B, T, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    normal = torch.randn(B, T, 3)
    albedo = torch.rand(B, T, 3)
    return position, direction, normal, albedo

def test_positional_encoding_shape_and_identity_at_zero_freqs():
    x = torch.randn(2, 5, 3)
    out0 = positional_encoding(x, n_freqs=0)
    assert torch.equal(out0, x)  # 0 частот = identity (только исходный x)

    out2 = positional_encoding(x, n_freqs=2)
    assert out2.shape == (2, 5, 3 * (1 + 2 * 2))
    # sin(0)=0, cos(0)=1 -- граничная проверка на x=0
    zeros = torch.zeros(1, 1, 3)
    enc = positional_encoding(zeros, n_freqs=3)
    expected = torch.cat([zeros] + [torch.zeros(1, 1, 3), torch.ones(1, 1, 3)] * 3, dim=-1)
    assert torch.allclose(enc, expected)


def test_spatial_dim_matches_actual_output_shape():
    position, direction, normal, albedo = _toy_vecs()
    for nfp, nfd in [(4, 2), (0, 0), (1, 8)]:
        out = build_spatial_conditioning(position, direction, normal, albedo,
                                          n_freqs_position=nfp, n_freqs_direction=nfd)
        assert out.shape[-1] == spatial_dim(n_freqs_position=nfp, n_freqs_direction=nfd)


def test_spatial_dim_with_roughness():
    position, direction, normal, albedo = _toy_vecs()
    roughness = torch.rand(*position.shape[:2], 1)
    out = build_spatial_conditioning(position, direction, normal, albedo, roughness=roughness)
    assert out.shape[-1] == spatial_dim(include_roughness=True)
    assert out.shape[-1] == spatial_dim(include_roughness=False) + 1


def test_position_scale_normalizes_before_encoding():
    """Регрессионный тест на баг 2026-08-20 (см. чат): без нормализации positional encoding
    применялась к сырым мировым координатам (пол Blender-сцены до ±4) -> высокие частоты
    алиасировались (аргумент sin/cos до ~100 радиан), целевая функция становилась негладкой,
    и рост объёма train-данных в 4.5x эмпирически НЕ уменьшал деградацию nrc_honest_spatial/
    gru_honest_spatial. Фикс: position / position_scale ПЕРЕД encoding. Здесь проверяем
    математическое тождество build_spatial_conditioning(position, ..., position_scale=S) ==
    build_spatial_conditioning(position / S, ..., position_scale=1.0) — т.е. эффект
    масштабирования именно там, где он и должен быть (на входе в positional_encoding,
    ДО конкатенации с direction/normal/albedo, которые position_scale не затрагивает)."""
    position, direction, normal, albedo = _toy_vecs()
    S = 4.0
    scaled_inline = build_spatial_conditioning(position, direction, normal, albedo,
                                                position_scale=S)
    pre_divided = build_spatial_conditioning(position / S, direction, normal, albedo,
                                              position_scale=1.0)
    assert torch.allclose(scaled_inline, pre_divided)

    # Без нормализации (position_scale=1.0, дефолт) сырые координаты сцены (±4) дают
    # положения encoding'а с частым "wrap-around" -- после нормализации (position_scale=4.0)
    # диапазон схлопывается к [-1,1], где высшая частота (2^3*pi≈25.1 при n_freqs_position=4)
    # даёт аргумент до ~25 рад (несколько оборотов, не десятки) -- не абсолютное отсутствие
    # алиасинга, но кардинально мягче. Проверяем хотя бы что результат отличается заметно.
    unnormalized = build_spatial_conditioning(position * 4.0, direction, normal, albedo,
                                               position_scale=1.0)
    normalized = build_spatial_conditioning(position * 4.0, direction, normal, albedo,
                                             position_scale=4.0)
    assert not torch.allclose(unnormalized, normalized)


def test_build_spatial_conditioning_batch_time_preserved():
    position, direction, normal, albedo = _toy_vecs(B=4, T=9)
    out = build_spatial_conditioning(position, direction, normal, albedo)
    assert out.shape[0] == 4 and out.shape[1] == 9
