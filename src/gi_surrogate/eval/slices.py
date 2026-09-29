"""Метрики по срезам (disocclusion/stable, dark/bright, расстояние до train).

Функции из evaluation/metrics.py без изменений; slice_mse — из dense_batch_core.py.
"""
import numpy as np


def disocclusion_vs_stable_mse(
    pred: np.ndarray, true: np.ndarray, disocclusion_flag: np.ndarray
) -> dict:
    """§6 delta — MSE раздельно для disocclusion vs non-disocclusion пикселей
    (флаг §2.5), ОРТОГОНАЛЬНО к static/step/drift (per_segment_type_mse) —
    именно этот срез определяет исход §5 stage1b (три возможных исхода),
    не агрегат по всему кадру.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    flag = np.asarray(disocclusion_flag).astype(bool)
    if pred.shape != flag.shape:
        raise ValueError(
            f"pred/disocclusion_flag shape mismatch: {pred.shape} vs {flag.shape}"
        )
    sq_err = (pred - true) ** 2
    stable = ~flag
    return dict(
        disocclusion=float(sq_err[flag].mean()) if flag.any() else float("nan"),
        stable=float(sq_err[stable].mean()) if stable.any() else float("nan"),
    )


def dark_vs_bright_mse(
    pred: np.ndarray, true: np.ndarray, dark_threshold: float
) -> dict:
    """TZ_stage8 доп. диагностика (13.09.2026, чат) — MSE раздельно для
    тёмных/светлых кадров, срез по true (luminance) относительно
    dark_threshold, вычисленного один раз на eval-множество сида (см.
    run_capacity_sweep_zeroday_kan.py). ОРТОГОНАЛЬНО disocclusion_vs_
    stable_mse (та делит по геометрическому событию, эта — по
    радиометрической яркости) и НЕ входит в пре-регистрированный §4
    decision rule TZ_stage8 — отдельная диагностика поверх него, не
    замена (изменение состава метрик задним числом после
    пре-регистрации иначе было бы p-hacking, см. TZ_stage8 §4).

    Мотивация: relative MC-шум растёт при низком радиансе (тот же
    физический эффект, из-за которого NRCRelativeL2Loss нормирует по
    предсказанию, а не по target, models/losses.py) — если KAN
    чувствительнее к шуму (arXiv 2407.14882), эффект должен быть
    сильнее именно в тёмных кадрах; срез по disocclusion/stable этого
    отдельно не покажет.

    pred, true: одной формы — одна последовательность/канал (см. вызов
    в _slice_mse_by_brightness).
    dark_threshold: скаляр — порог luminance; "тёмный" = true < threshold.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"pred/true shape mismatch: {pred.shape} vs {true.shape}")
    sq_err = (pred - true) ** 2
    dark = true < dark_threshold
    bright = ~dark
    return dict(
        dark=float(sq_err[dark].mean()) if dark.any() else float("nan"),
        bright=float(sq_err[bright].mean()) if bright.any() else float("nan"),
    )


def compute_age_since_disocclusion(disocclusion_flag: np.ndarray) -> np.ndarray:
    """Возраст (в шагах последовательности, НЕ реальном времени — регулярный
    Δt дозволяет это, §2.3) с последнего disocclusion-события: 0 в самом
    событии (disocclusion_flag=1), растёт на 1 с каждым следующим стабильным
    шагом. NaN — до первого события в этой последовательности (если весь
    префикс стабилен без единого события, что не должно случаться на
    практике: idx=0 всегда disocclusion_flag=1 по конвенции cold-start, см.
    blender/dataset_adapter_dense.py).

    Принимает 1D [T] (одна последовательность) ИЛИ 2D [B, T] (батч
    независимых последовательностей — возраст считается ОТДЕЛЬНО по каждой
    строке, события одного пикселя не влияют на другой).
    """
    flag = np.asarray(disocclusion_flag).astype(bool)
    if flag.ndim == 1:
        age = np.full(flag.shape, np.nan, dtype=np.float64)
        last_event = -1
        for i, f in enumerate(flag):
            if f:
                last_event = i
                age[i] = 0.0
            elif last_event >= 0:
                age[i] = float(i - last_event)
        return age
    if flag.ndim == 2:
        return np.stack([compute_age_since_disocclusion(row) for row in flag], axis=0)
    raise ValueError(f"disocclusion_flag должен быть 1D или 2D, получено ndim={flag.ndim}")


def error_vs_warp_age_curve(
    pred: np.ndarray,
    true: np.ndarray,
    disocclusion_flag: np.ndarray,
    bin_edges: np.ndarray | None = None,
) -> dict:
    """§6 delta — аналог error_vs_offset_curve (§6.1), но событие-триггер —
    disocclusion (§2.5), а не световой скачок light_schedule: "error как
    функция возраста истории после warp (сколько кадров прошло с последней
    валидной диссокклюзии в этом пикселе)".

    pred/true/disocclusion_flag — одной формы, 1D [T] или 2D [B, T] (батч
    пикселей, см. compute_age_since_disocclusion). Точки БЕЗ определённого
    возраста (NaN — до первого события) исключаются из кривой, тем же
    принципом, что error_vs_offset_curve исключает offset<0/NaN.
    """
    if bin_edges is None:
        bin_edges = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, np.inf])

    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    age = compute_age_since_disocclusion(disocclusion_flag).reshape(-1)
    if pred.shape != age.shape:
        raise ValueError(f"pred/disocclusion_flag shape mismatch after flatten: {pred.shape} vs {age.shape}")

    mask = ~np.isnan(age)
    sq_err = (pred[mask] - true[mask]) ** 2
    a = age[mask]

    bin_idx = np.digitize(a, bin_edges[1:-1])
    curve = {}
    for b in range(len(bin_edges) - 1):
        b_mask = bin_idx == b
        label = f"[{bin_edges[b]:g},{bin_edges[b+1]:g})"
        curve[label] = float(sq_err[b_mask].mean()) if b_mask.any() else float("nan")
    return curve


def error_vs_distance_to_train_curve(
    pred: np.ndarray,
    true: np.ndarray,
    eval_pixel_coords: np.ndarray,
    train_pixel_coords: np.ndarray,
    bin_edges: np.ndarray | None = None,
) -> dict:
    """07.09.2026, чат — механистический аналог error_vs_warp_age_curve, но
    ось не время, а ПРОСТРАНСТВО: ошибка как функция евклидова расстояния (в
    пикселях экрана) от eval-точки до ближайшей train-точки.

    Мотивация: агрегатный disocc/stable MSE не чувствителен к дырам в
    пространственном покрытии train-точек (см. mempalace/EXPERIMENT_LOG.md
    07.09.2026, продакшн-прогон tile_stratified_sample на Zero Day — MSE
    почти не сдвинулся). Эта кривая — прямая проверка гипотезы "модель хуже
    экстраполирует туда, где обучающих точек рядом не было", а также способ
    отличить "sample density зависит от координаты в пространстве" (реальный
    эффект) от "просто нужно больше точек вообще" (эффект объёма данных,
    ортогональный расположению).

    pred, true: [N_eval, ...] -- одна запись на eval-точку, произвольное
        число хвостовых осей (T, C — время, каналы, как в остальном проекте,
        см. _slice_mse в run_hybrid_v0.py). Квадратичная ошибка усредняется
        по ВСЕМ хвостовым осям сразу (не по T потом по C раздельно, как
        _slice_mse) -- здесь не нужна отдельная точность по каналам, только
        один скаляр на eval-точку для биннинга по расстоянию.
    eval_pixel_coords: [N_eval, 2] -- (x, y) для каждой eval-точки, тот же
        порядок, что и pred/true по оси 0.
    train_pixel_coords: [N_train, 2] -- (x, y) реально использованных
        train-точек этого сида (после tile_stratified_sample/perm[:n]).
    bin_edges: границы бинов расстояния в пикселях экрана. По умолчанию
        [0,5,10,20,30,50,80,120,inf) -- подобрано под масштаб Zero Day
        (320x180, ~90 точек -> характерный шаг тайла ~sqrt(320*180/90)~25px,
        так что дефолтные бины покрывают "внутри своего тайла" через
        "далеко от любой train-точки"). Для другого H,W/бюджета бины стоит
        передавать явно.

    Возвращает dict той же формы, что error_vs_warp_age_curve/
    error_vs_offset_curve (label -> mean squared error, NaN для пустых
    бинов) -- пустой бин ЗНАЧИТ, что при данном бюджете точек ни одна
    eval-точка не оказалась на этом расстоянии от ближайшей train-точки
    (сам по себе диагностический факт про покрытие, не только про ошибку).

    Использует scipy.spatial.cKDTree для поиска ближайшего сосуда -- N_eval
    может быть большим (57600 на Zero Day), брутфорс O(N_eval*N_train) для
    большого train-бюджета был бы дороже без явной необходимости.
    """
    from scipy.spatial import cKDTree

    if bin_edges is None:
        bin_edges = np.array([0, 5, 10, 20, 30, 50, 80, 120, np.inf])

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"pred/true shape mismatch: {pred.shape} vs {true.shape}")
    n_eval = pred.shape[0]
    sq_err = ((pred - true) ** 2).reshape(n_eval, -1).mean(axis=1)

    eval_pixel_coords = np.asarray(eval_pixel_coords, dtype=np.float64)
    train_pixel_coords = np.asarray(train_pixel_coords, dtype=np.float64)
    if eval_pixel_coords.shape != (n_eval, 2):
        raise ValueError(
            f"eval_pixel_coords должен быть [{n_eval}, 2], получено {eval_pixel_coords.shape}"
        )

    tree = cKDTree(train_pixel_coords)
    dist, _ = tree.query(eval_pixel_coords, k=1)

    bin_idx = np.digitize(dist, bin_edges[1:-1])
    curve = {}
    for b in range(len(bin_edges) - 1):
        b_mask = bin_idx == b
        label = f"[{bin_edges[b]:g},{bin_edges[b+1]:g})"
        curve[label] = float(sq_err[b_mask].mean()) if b_mask.any() else float("nan")
    return curve


def slice_mse(preds_kind, true, disocc_flags, n_eval):
    """Идентично run_hybrid_v0.py::_slice_mse (переименовано без ведущего
    подчёркивания — здесь публичная функция модуля, не приватный helper
    внутри чужого файла)."""
    disocc_vals, stable_vals = [], []
    for p in range(n_eval):
        pred_p = preds_kind[p].astype(np.float64)
        true_p = true[p].astype(np.float64)
        flag_p = disocc_flags[p].astype(bool)
        d_c, s_c = [], []
        for c in range(pred_p.shape[-1]):
            split = disocclusion_vs_stable_mse(pred_p[:, c], true_p[:, c], flag_p)
            if not np.isnan(split["disocclusion"]):
                d_c.append(split["disocclusion"])
            if not np.isnan(split["stable"]):
                s_c.append(split["stable"])
        if d_c:
            disocc_vals.append(float(np.mean(d_c)))
        if s_c:
            stable_vals.append(float(np.mean(s_c)))
    return (float(np.mean(disocc_vals)) if disocc_vals else float("nan"),
            float(np.mean(stable_vals)) if stable_vals else float("nan"))


def slice_mse_by_brightness(preds_kind, true, n_eval, dark_threshold):
    """Аналог _slice_mse (run_hybrid_v0.py), но срез по яркости (dark_vs_
    bright_mse, evaluation/metrics.py), не по disocclusion_flag — доп.
    диагностика, добавлена по ходу разговора 13.09.2026 (см. TZ_stage8,
    addendum к §3.8), НЕ входит в пре-регистрированный §4 decision rule."""
    dark_vals, bright_vals = [], []
    for p in range(n_eval):
        pred_p = preds_kind[p].astype(np.float64)
        true_p = true[p].astype(np.float64)
        d_c, b_c = [], []
        for c in range(pred_p.shape[-1]):
            split = dark_vs_bright_mse(pred_p[:, c], true_p[:, c], dark_threshold)
            if not np.isnan(split["dark"]):
                d_c.append(split["dark"])
            if not np.isnan(split["bright"]):
                b_c.append(split["bright"])
        if d_c:
            dark_vals.append(float(np.mean(d_c)))
        if b_c:
            bright_vals.append(float(np.mean(b_c)))
    return (float(np.mean(dark_vals)) if dark_vals else float("nan"),
            float(np.mean(bright_vals)) if bright_vals else float("nan"))
