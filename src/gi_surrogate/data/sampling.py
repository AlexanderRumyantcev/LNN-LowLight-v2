"""
tile_stratified_sample -- выбор небольшого числа обучающих точек из уже
готового плотного per-pixel рендера с гарантией равномерного покрытия
экрана, вместо плоского np.random.permutation(...)[:n].

07.09.2026, чат, TZ_stage5 (продолжение). Мотивация (см. mempalace,
drawer f7628c02...): на Zero Day (320x180=57600 валидных пикселей)
train_idx выбирался как perm[:90] -- плоский случайный permutation без
пространственной стратификации. Это НЕ гарантирует покрытие экрана:
чисто по вероятности 90 точек могут кучковаться, оставляя целые области
кадра без единой обучающей точки -- и после добавления spatial
conditioning (models/spatial_features.py) это стало реальной проблемой
(модель не видела геометрию в непокрытых областях вообще).

Перенесённый принцип -- Müller et al. 2021 (NRC), §3.5 "Amortization in
a Real-time Path Tracer", пункт про tile-based training-path selection:
экран делится на тайлы, из каждого тайла берётся ОДНА точка со случайным
сдвигом внутри тайла -- "uniform sparse set... in screen space" вместо
регулярной сетки или чистой случайности. У NRC это привязано к
адаптивному per-frame бюджету (decoupled от разрешения живого рендера);
здесь адаптивная часть НЕ переносится -- см. обсуждение в чате
(07.09.2026) о том, что она решает другую задачу (бюджет обучения в
реальном времени), не применимую к офлайн-датасету. Переносится только
сам паттерн размещения точек.

НЕ привязано к Zero Day конкретно -- работает с любым списком
последовательностей вида [{"pixel": (x, y), ...}, ...] (та же схема,
что build_all_pixel_sequences/build_all_pixel_sequences_zeroday), в
т.ч. потенциально пригодится для dt1.npz/dt1_geomv2.npz (там сейчас та
же схема perm[:n_train_pixels] в run_experiment_dense.py -- не тронуто
в этом изменении, см. mempalace: сфокусировано на Zero Day, это была
явно заявленная граница задачи в этой сессии).
"""
import numpy as np


def tile_stratified_sample(seqs, n_samples, rng, H=None, W=None):
    """Выбирает n_samples индексов в seqs, стратифицированных по тайлам
    экрана (NRC §3.5: тайл + случайный сдвиг на тайл).

    seqs: список dict, каждый со своим ключом "pixel" = (x, y) --
        координата данной последовательности в кадре. Порядок в списке
        не имеет значения (в отличие от плоского permutation, здесь НЕ
        предполагается raster-порядок и НЕ ломается, если часть
        пикселей отфильтрована как невалидная -- ровно случай
        load_fully_valid_pixel_sequences_zeroday, где часть кадра может
        быть исключена).
    n_samples: сколько точек вернуть (обучающий бюджет, аналог NRC's
        training record budget -- здесь фиксированный, не адаптивный,
        см. докстринг модуля).
    rng: np.random.Generator (например np.random.default_rng(seed)) --
        сид передаётся явно вызывающим кодом, тем же принципом, что
        везде в проекте (per-seed воспроизводимость).
    H, W: размеры кадра. Если не заданы -- выводятся из максимальных
        координат в seqs (+1). Это приближение (края кадра могут быть
        занижены, если крайний пиксель невалиден на всех кадрах) --
        осознанный компромисс ради независимости функции от отдельного
        чтения .npz только за размерами (см. обсуждение в чате).

    Возвращает: list[int] длиной n_samples -- индексы в seqs (не
    пиксельные координаты). Если валидных пикселей меньше n_samples,
    возвращает все доступные (без дублей).

    Алгоритм:
    1. Строит H x W карту "индекс в seqs или -1" по полю pixel.
    2. Делит кадр на сетку тайлов (~n_samples тайлов, с учётом aspect
       ratio W/H, чтобы тайлы были примерно квадратными).
    3. Из каждого тайла берёт ОДИН случайный присутствующий в seqs
       пиксель (случайный "сдвиг" внутри тайла); пустые тайлы (без
       валидных пикселей) пропускаются, не добиваются искусственно.
    4. Если после прохода всех тайлов набралось меньше n_samples точек
       (пустые тайлы или n_tiles < n_samples из-за округления) --
       добор случайными валидными пикселями вне уже выбранных, чтобы
       итоговый train-бюджет не менялся по сравнению со старой схемой.
    """
    if len(seqs) == 0:
        return []
    n_samples = min(n_samples, len(seqs))

    xs = np.fromiter((s["pixel"][0] for s in seqs), dtype=np.int64, count=len(seqs))
    ys = np.fromiter((s["pixel"][1] for s in seqs), dtype=np.int64, count=len(seqs))
    if W is None:
        W = int(xs.max()) + 1
    if H is None:
        H = int(ys.max()) + 1

    idx_grid = np.full((H, W), -1, dtype=np.int64)
    idx_grid[ys, xs] = np.arange(len(seqs), dtype=np.int64)

    # Число тайлов по каждой оси ~ sqrt(n_samples), скорректировано под aspect
    # ratio кадра, чтобы тайлы были примерно квадратными (не вытянутыми) -- та
    # же логика, что и у "почти квадратных" тайлов NRC (§3.5, там квадратные
    # тайлы фиксированного пиксельного размера; здесь размер подстраивается
    # под H,W заранее известного плотного рендера, а не под живой кадр).
    aspect = W / H
    n_tiles_y = max(1, round((n_samples / aspect) ** 0.5))
    n_tiles_x = max(1, round(n_samples / n_tiles_y))

    y_edges = np.linspace(0, H, n_tiles_y + 1)
    x_edges = np.linspace(0, W, n_tiles_x + 1)

    chosen = []
    chosen_set = set()
    tile_order = [(ty, tx) for ty in range(n_tiles_y) for tx in range(n_tiles_x)]
    rng.shuffle(tile_order)  # порядок обхода тайлов тоже случаен -- при
    # нехватке бюджета (n_tiles > n_samples) не отдаём предпочтение верхним
    # тайлам систематически
    for ty, tx in tile_order:
        if len(chosen) >= n_samples:
            break
        y0, y1 = int(y_edges[ty]), int(y_edges[ty + 1])
        x0, x1 = int(x_edges[tx]), int(x_edges[tx + 1])
        if y1 <= y0 or x1 <= x0:
            continue
        tile = idx_grid[y0:y1, x0:x1]
        valid_in_tile = tile[tile >= 0]
        if valid_in_tile.size == 0:
            continue
        pick = int(valid_in_tile[rng.integers(valid_in_tile.size)])
        if pick not in chosen_set:
            chosen.append(pick)
            chosen_set.add(pick)

    if len(chosen) < n_samples:
        remaining = np.array([i for i in range(len(seqs)) if i not in chosen_set], dtype=np.int64)
        extra_n = n_samples - len(chosen)
        if remaining.size > 0:
            extra = rng.choice(remaining, size=min(extra_n, remaining.size), replace=False)
            chosen.extend(int(i) for i in extra)

    return chosen[:n_samples]
