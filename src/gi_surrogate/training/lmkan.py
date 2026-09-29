"""Обучение и инференс lmKAN (fused CUDA-кернелы) с hessian-регуляризацией и decay-расписанием.

Вынесено без изменений логики из run_tz_stage11_stage1_isocompute_accuracy.py (TZ_stage11).
"""
import torch
from gi_surrogate.data.batch import SPATIAL_DIM, get_spatial
from gi_surrogate.models.kan_lmkan import NRCLmKANBaseline
from gi_surrogate.models.losses import NRCRelativeL2Loss

NUM_GRIDS = 10                  # README lmkan / models/kan_lmkan_baseline.py, проектная константа

TILE_SIZE_FORWARD = 8           # models/kan_lmkan_baseline.py дефолт

KAN_N_HIDDEN_LAYERS = 5         # структурное соответствие nrc_strong (TZ_stage8 §3.2, тот же принцип)

# HESS_LAMBDA_INIT/FINAL — порядок величины, НЕ тюнятся под точность (тот же
# принцип, что CURVATURE_LAMBDA в TZ_stage8 §3.4) — авторы lmKAN (README)
# рекомендуют "начать с сильной регуляризации, ослаблять по ходу обучения",
# без конкретных чисел; 1e-2 -> 1e-4 даёт тот же порядок эффекта на старте,
# что CURVATURE_LAMBDA=1e-3 в TZ_stage8 (среднее между init/final), и уходит
# на 2 порядка к концу обучения — не более того обосновано на этом этапе.
HESS_LAMBDA_INIT = 1e-2

HESS_LAMBDA_FINAL = 1e-4


def hess_lambda_schedule(epoch, total_epochs, lambda_init, lambda_final):
    """Геометрическая (лог-линейная) интерполяция между lambda_init (epoch=0)
    и lambda_final (epoch=total_epochs-1) — "сильная регуляризация в начале,
    ослабление по ходу обучения" (README lmkan, ТЗ §2.3), без дополнительной
    формы (cosine/step), которая не была бы обоснована на этом этапе."""
    if total_epochs <= 1:
        return lambda_init
    frac = epoch / (total_epochs - 1)
    return float(lambda_init * (lambda_final / lambda_init) ** frac)


def train_nrc_lmkan(hidden_dim, batch, device, epochs, lr,
                      hess_lambda_init, hess_lambda_final):
    """Аналог train_nrc_kan (TZ_stage8, run_capacity_sweep_zeroday_kan.py),
    но модель — NRCLmKANBaseline (fused CUDA lmKAN, TZ_stage9/models/kan_
    lmkan_baseline.py), и hessian_regularization_loss с decay-schedule
    (ТЗ §2.3, ранее намеренно не использовалась в TZ_stage9, см. докстринг
    hessian_regularization_loss в models/kan_lmkan_baseline.py).

    Возвращает (model, diverged: bool, last_data_loss, last_hess_loss).
    Если loss становится NaN на каком-то эпохе — обучение останавливается
    НЕМЕДЛЕННО (веса после этого непригодны для инференса, ТЗ §2.4), модель
    всё равно возвращается (для диагностики числа параметров), но diverged=True
    сигнализирует вызывающему коду не использовать её предсказания в
    массивах метрик.
    """
    obs = batch["obs"].to(device)
    cold = batch["cold"].to(device)
    conf = batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    obs_dim = obs.shape[-1]

    spatial = get_spatial(batch, device)
    model = NRCLmKANBaseline(
        obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
        staleness_dim=4, spatial_dim=SPATIAL_DIM,
        n_hidden_layers=KAN_N_HIDDEN_LAYERS, num_grids=NUM_GRIDS,
    ).to(device)
    u = NRCLmKANBaseline.build_input(obs, cold, conf, use_staleness=True,
                                      extra_staleness=[disocc, mismatch], spatial=spatial)
    true = batch["true"].to(device)
    loss_fn = NRCRelativeL2Loss(use_confidence_weight=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    diverged = False
    data_loss_last, hess_loss_last = None, None
    for ep in range(epochs):
        optimizer.zero_grad()
        pred = model(u)
        data_loss = loss_fn(pred, true)
        hess_loss = model.hessian_regularization_loss()
        hess_lambda = hess_lambda_schedule(ep, epochs, hess_lambda_init, hess_lambda_final)
        loss = data_loss + hess_lambda * hess_loss
        if not torch.isfinite(loss):
            diverged = True
            print(f"    [DIVERGED at epoch {ep}] loss={float(loss) if loss == loss else 'nan'} "
                  f"data_loss={float(data_loss) if data_loss == data_loss else 'nan'} "
                  f"hess_loss={float(hess_loss) if hess_loss == hess_loss else 'nan'}",
                  flush=True)
            break
        loss.backward()
        optimizer.step()
        if ep == epochs - 1:
            data_loss_last, hess_loss_last = float(data_loss.item()), float(hess_loss.item())
    model._last_data_loss = data_loss_last
    model._last_hess_loss = hess_loss_last
    model._diverged = diverged
    return model, diverged


def predict_nrc_lmkan(model, batch, device):
    obs = batch["obs"].to(device)
    cold, conf = batch["cold"].to(device), batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    with torch.no_grad():
        spatial = get_spatial(batch, device)
        u = NRCLmKANBaseline.build_input(obs, cold, conf, use_staleness=True,
                                          extra_staleness=[disocc, mismatch], spatial=spatial)
        return model(u).cpu().numpy()
