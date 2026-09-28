"""Статистика сравнения архитектур: paired bootstrap и seed-robustness.

Перенесено без изменений из evaluation/metrics.py.
"""
import numpy as np

MIN_N_SEEDS = 8               # §6.4 — n=3 признан недостаточным в spike-тесте


def paired_bootstrap_significance(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    §6.4 — paired bootstrap ПО ПОСЛЕДОВАТЕЛЬНОСТЯМ (errors_a/errors_b — один скаляр на
    последовательность/probe, не на отдельный шаг: шаги внутри последовательности
    коррелированы, ресэмплинг по ним нарушил бы предположение о независимости).

    errors_a, errors_b: [N_seeds] — например per-segment-type MSE или early-zone MSE,
        посчитанные отдельно на каждой из N_seeds независимых последовательностей/проб.

    Returns: dict(mean_diff, ci_low, ci_high, n_seeds, significant)
        significant = True, если 95% CI разницы (a - b) не содержит 0.
    """
    errors_a = np.asarray(errors_a, dtype=np.float64)
    errors_b = np.asarray(errors_b, dtype=np.float64)
    if errors_a.shape != errors_b.shape:
        raise ValueError("errors_a и errors_b должны быть одной формы (paired)")
    n_seeds = len(errors_a)
    if n_seeds < MIN_N_SEEDS:
        raise ValueError(
            f"n_seeds={n_seeds} < {MIN_N_SEEDS} — §6.4: на n=3 оценки значимости "
            f"были нестабильны в spike-тесте, минимум {MIN_N_SEEDS}."
        )

    rng = np.random.default_rng(seed)
    diff = errors_a - errors_b
    mean_diff = float(diff.mean())

    resampled_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n_seeds, size=n_seeds)  # paired resampling — тот же idx на a и b
        resampled_means[i] = diff[idx].mean()

    alpha = 1.0 - ci
    lo, hi = np.quantile(resampled_means, [alpha / 2, 1.0 - alpha / 2])
    significant = not (lo <= 0.0 <= hi)

    return dict(
        mean_diff=mean_diff, ci_low=float(lo), ci_high=float(hi),
        n_seeds=n_seeds, significant=significant,
    )


def seed_robustness_report(
    errors_a: np.ndarray, errors_b: np.ndarray,
) -> dict:
    """
    Диагностика устойчивости вывода к выбору набора сидов — ДОПОЛНЕНИЕ к
    paired_bootstrap_significance, не замена. Мотивация (сессия 2026-08-23):
    CI paired-bootstrap говорит "задевает ли разница ноль", но не говорит,
    насколько эффект велик относительно межсидового шума — а это отдельный
    вопрос, который на n_seeds=MIN_N_SEEDS=8 может быть некритично мал.
    Рекомендация читать std между сидами против размера эффекта — практика
    из статистического приложения DeepONet (Lu et al. 2021) и из общего
    анализа мощности в Colas & Sigaud ("How Many Random Seeds?", 2018):
    статистически значимый на конкретном n эффект может быть "хрупким",
    если |Cohen's d| мал — тогда для уверенности нужен стресс-тест на
    большем n_seeds, а не просто больше decimal places в CI.

    errors_a, errors_b: [N_seeds] — тот же формат paired-массивов, что и
        в paired_bootstrap_significance (один скаляр на сид/probe).

    Cohen's d здесь — mean(diff)/std(diff, ddof=1) для paired-разницы
    (не независимая version формулы) — грубый ориентир (Cohen 1988:
    0.2/0.5/0.8 = малый/средний/большой), НЕ формальный тест мощности.
    """
    a = np.asarray(errors_a, dtype=np.float64)
    b = np.asarray(errors_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("errors_a и errors_b должны быть одной формы (paired)")
    diff = a - b
    n = len(diff)
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if n > 1 else float("nan")
    cohens_d = mean_diff / std_diff if std_diff > 0 else float("nan")

    if np.isnan(cohens_d):
        recommendation = "std_diff=0 — вырожденный случай, проверить входные данные"
    elif abs(cohens_d) >= 0.8:
        recommendation = "большой эффект (|d|>=0.8) — вывод устойчив, доп. сиды маловероятно поменяют знак"
    elif abs(cohens_d) >= 0.5:
        recommendation = f"средний эффект (0.5<=|d|<0.8) при n_seeds={n} — вероятно устойчив, но стоит перепроверить на большем n (Colas & Sigaud)"
    else:
        recommendation = f"МАЛЫЙ эффект (|d|<0.5) при n_seeds={n} — значимость может быть хрупкой, перепроверить на большем n (Colas & Sigaud) ОБЯЗАТЕЛЬНО перед архитектурными выводами"

    return dict(
        mean_diff=mean_diff, std_diff=std_diff, cohens_d=cohens_d,
        n_seeds=n, recommendation=recommendation,
    )
