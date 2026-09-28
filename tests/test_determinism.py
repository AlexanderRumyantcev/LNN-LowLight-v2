"""Тесты determinism.py (set_determinism, capture_env_metadata, get_backend_name).

Перенесены из tests/test_determinism_and_cross_env.py; тесты cross-env протокола
run_experiment_dense (_compare_significance_dicts, run_cross_env_check) остались в архиве.
"""
import numpy as np
import torch

from gi_surrogate.utils.determinism import set_determinism, capture_env_metadata, get_backend_name

def test_set_determinism_sets_global_flags():
    set_determinism(seed=0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_set_determinism_same_seed_gives_same_numbers():
    set_determinism(seed=42)
    a = torch.randn(5)
    set_determinism(seed=42)
    b = torch.randn(5)
    assert torch.equal(a, b), "одинаковый seed должен давать побитово одинаковый результат"


def test_capture_env_metadata_reads_backend_from_given_device():
    meta_cpu = capture_env_metadata(torch.device("cpu"))
    assert meta_cpu["backend"] == "cpu"
    assert set(meta_cpu.keys()) == {
        "backend", "torch_version", "cuda_version", "cudnn_version",
        "python_version", "hostname", "git_commit",
    }
    # регрессия на баг первой версии: backend должен браться из ПЕРЕДАННОГО device,
    # а не из os.environ.get("_FORCE_CPU_KINDS") (в реальном коде это python-множество
    # имён моделей, а не env-переменная — см. mempalace)
    if torch.backends.mps.is_available():
        meta_mps = capture_env_metadata(torch.device("mps"))
        assert meta_mps["backend"] == "mps"


def test_get_backend_name_without_device_returns_valid_string():
    assert get_backend_name() in ("cpu", "mps", "cuda")


# --- модуль 2: cross-environment protocol (логика сравнения, без обучения) --

def _sig(mean_diff, significant):
    return {"mean_diff": mean_diff, "significant": significant, "ci_low": 0.0,
            "ci_high": 0.0, "n_seeds": 8}
