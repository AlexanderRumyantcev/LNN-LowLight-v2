"""Пути репозитория: данные вне git, референсные результаты - в git.

REPO_ROOT определяется от расположения этого файла, поэтому пакет нужно запускать
из исходников (PYTHONPATH=src или pip install -e .), а не из обычной установки.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"                       # *.npz, в git не хранится (на Kaggle - --dataset)
REFERENCE_DIR = REPO_ROOT / "reference_results"     # небольшие JSON готовых результатов, в git
OUTPUT_DIR = REPO_ROOT / "outputs"                  # результаты локальных прогонов, в git не хранится (.gitignore)

DEFAULT_DATASET = DATA_DIR / "generate_dataset_zeroday_result_nostep_v1.npz"
# 16-сидовые nrc_strong@{16,64} из TZ_stage8: MLP-референс, не переобучается (TZ_stage11 §2.3)
TZ8_KAN_RESULTS = REFERENCE_DIR / "run_capacity_sweep_zeroday_kan_results.json"
# гипотеза 1 (weight_decay для nrc_strong@64) — вход для hypothesis1b (сравнение подходов)
HYPOTHESIS1_RESULTS = REFERENCE_DIR / "run_hypothesis1_reg_nrc_strong_results.json"


def resolve_out_path(filename, out_dir=None):
    """Куда писать результат прогона. НЕ рядом с датасетом (Path(dataset_path).parent) —
    на Kaggle это /kaggle/input/..., read-only (см. чат 25-26.09.2026, реальный
    OSError [Errno 30] на run_tz_stage9_online_speed.py / stage11 stage0/stage1).
    Порядок: явный --out_dir -> /kaggle/working (если есть) -> OUTPUT_DIR локально."""
    if out_dir:
        base = Path(out_dir)
    elif Path("/kaggle/working").exists():
        base = Path("/kaggle/working")
    else:
        base = OUTPUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / filename
