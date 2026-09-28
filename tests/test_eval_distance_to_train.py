import sys
from pathlib import Path

import numpy as np
import pytest

from gi_surrogate.eval.slices import error_vs_distance_to_train_curve


def test_zero_error_at_train_points_themselves():
    """Eval-точки, совпадающие с train-точками, попадают в bin [0,5) с
    заведомо малой ошибкой (тривиальная санити-проверка биннинга)."""
    eval_coords = np.array([[0.0, 0.0], [100.0, 100.0]])
    train_coords = np.array([[0.0, 0.0], [100.0, 100.0]])
    pred = np.zeros((2, 4, 3))
    true = np.zeros((2, 4, 3))
    curve = error_vs_distance_to_train_curve(pred, true, eval_coords, train_coords)
    assert curve["[0,5)"] == 0.0


def test_far_points_land_in_far_bin():
    eval_coords = np.array([[0.0, 0.0], [500.0, 500.0]])
    train_coords = np.array([[0.0, 0.0]])
    pred = np.array([[1.0], [1.0]])
    true = np.array([[0.0], [0.0]])
    curve = error_vs_distance_to_train_curve(pred, true, eval_coords, train_coords)
    # первая точка (dist=0) в ближнем бине, вторая (dist~707) в дальнем
    assert curve["[0,5)"] == 1.0
    assert curve["[120,inf)"] == 1.0
    assert np.isnan(curve["[10,20)"])  # пустой бин


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        error_vs_distance_to_train_curve(
            np.zeros((3, 2)), np.zeros((3, 2)),
            np.zeros((2, 2)), np.zeros((1, 2)),
        )


def test_custom_bin_edges():
    eval_coords = np.array([[0.0, 0.0], [50.0, 0.0]])
    train_coords = np.array([[0.0, 0.0]])
    pred = np.array([2.0, 4.0])
    true = np.array([0.0, 0.0])
    curve = error_vs_distance_to_train_curve(
        pred, true, eval_coords, train_coords, bin_edges=np.array([0, 25, np.inf]),
    )
    assert set(curve.keys()) == {"[0,25)", "[25,inf)"}
    assert curve["[0,25)"] == 4.0
    assert curve["[25,inf)"] == 16.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
