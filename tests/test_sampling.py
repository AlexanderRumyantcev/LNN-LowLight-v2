import sys
from pathlib import Path

import numpy as np
import pytest

from gi_surrogate.data.sampling import tile_stratified_sample


def _make_dense_seqs(H, W):
    return [{"pixel": (x, y)} for y in range(H) for x in range(W)]


def test_returns_exact_count_and_unique():
    seqs = _make_dense_seqs(180, 320)
    rng = np.random.default_rng(0)
    idx = tile_stratified_sample(seqs, 90, rng)
    assert len(idx) == 90
    assert len(set(idx)) == 90
    assert all(0 <= i < len(seqs) for i in idx)


def test_deterministic_given_same_seed():
    seqs = _make_dense_seqs(180, 320)
    idx_a = tile_stratified_sample(seqs, 90, np.random.default_rng(42))
    idx_b = tile_stratified_sample(seqs, 90, np.random.default_rng(42))
    assert idx_a == idx_b


def test_different_seeds_differ():
    seqs = _make_dense_seqs(180, 320)
    idx_a = tile_stratified_sample(seqs, 90, np.random.default_rng(0))
    idx_b = tile_stratified_sample(seqs, 90, np.random.default_rng(1))
    assert idx_a != idx_b


def test_coverage_all_quadrants_hit():
    """Ядро мотивации: при 90 точках на 320x180 (~10x9 тайлов) каждый из 4
    квадрантов кадра должен получить хотя бы одну точку -- то, что плоский
    permutation НЕ гарантирует (см. докстринг модуля)."""
    H, W = 180, 320
    seqs = _make_dense_seqs(H, W)
    rng = np.random.default_rng(0)
    idx = tile_stratified_sample(seqs, 90, rng)
    xs = np.array([seqs[i]["pixel"][0] for i in idx])
    ys = np.array([seqs[i]["pixel"][1] for i in idx])
    mid_x, mid_y = W / 2, H / 2
    quadrants = [
        ((xs < mid_x) & (ys < mid_y)),
        ((xs >= mid_x) & (ys < mid_y)),
        ((xs < mid_x) & (ys >= mid_y)),
        ((xs >= mid_x) & (ys >= mid_y)),
    ]
    for q in quadrants:
        assert q.sum() >= 1


def test_max_gap_smaller_than_plain_permutation():
    """Статистическая проверка: у tile-stratified максимальный 'разрыв' между
    соседними точками по сетке тайлов меньше, чем у чистого permutation того
    же бюджета (усреднено по нескольким сидам, чтобы не быть флейки)."""
    H, W = 180, 320
    seqs = _make_dense_seqs(H, W)
    n_tiles_side = 6  # огрубляем до 6x6 контрольной сетки для подсчёта пустых клеток

    def empty_cells(idx):
        xs = np.array([seqs[i]["pixel"][0] for i in idx])
        ys = np.array([seqs[i]["pixel"][1] for i in idx])
        cell_x = (xs / W * n_tiles_side).astype(int).clip(max=n_tiles_side - 1)
        cell_y = (ys / H * n_tiles_side).astype(int).clip(max=n_tiles_side - 1)
        occupied = set(zip(cell_x.tolist(), cell_y.tolist()))
        return n_tiles_side * n_tiles_side - len(occupied)

    tile_empty, perm_empty = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        idx_tile = tile_stratified_sample(seqs, 30, rng)
        tile_empty.append(empty_cells(idx_tile))

        perm = np.random.default_rng(seed).permutation(len(seqs))[:30]
        perm_empty.append(empty_cells(list(perm)))

    assert np.mean(tile_empty) < np.mean(perm_empty)


def test_handles_holes_gracefully():
    """Часть пикселей отсутствует в seqs (как при фильтрации невалидных на
    Zero Day/dt1.npz) -- функция не должна падать и должна честно добрать
    бюджет из оставшихся, если целые тайлы оказались пустыми."""
    H, W = 40, 40
    seqs = [{"pixel": (x, y)} for y in range(H) for x in range(W) if (x + y) % 3 != 0]
    rng = np.random.default_rng(0)
    idx = tile_stratified_sample(seqs, 50, rng, H=H, W=W)
    assert len(idx) == 50
    assert len(set(idx)) == 50


def test_more_samples_than_seqs_returns_all():
    seqs = _make_dense_seqs(5, 5)
    rng = np.random.default_rng(0)
    idx = tile_stratified_sample(seqs, 1000, rng)
    assert len(idx) == len(seqs)
    assert set(idx) == set(range(len(seqs)))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
