"""Обучение и инференс KAN на efficient-kan (TZ_stage8).

Вынесено без изменений логики из run_capacity_sweep_zeroday_kan.py.
"""
import torch
from gi_surrogate.data.batch import SPATIAL_DIM, get_spatial
from gi_surrogate.models.kan_efficient import NRCKANBaseline
from gi_surrogate.models.losses import NRCRelativeL2Loss

KAN_N_HIDDEN_LAYERS = 5         # структурное соответствие nrc_strong (TZ_stage8 §3.2)

KAN_GRID_SIZE = 5               # efficient-kan дефолт, TZ_stage8 §3.2 — не тюнится

KAN_SPLINE_ORDER = 3            # efficient-kan дефолт, TZ_stage8 §3.2 — не тюнится

# TZ_stage8 §3.4: фиксированный порядок величины, НЕ подобран под точность на
# eval — калибровка только по масштабу (см. печать ratio в train-цикле ниже),
# пересмотр допустим лишь если диагностика покажет полное доминирование/
# незначимость (>=3 порядка разницы с data-loss), не как тюнинг под качество.
CURVATURE_LAMBDA = 1e-3


def train_nrc_kan(hidden_dim, batch, device, epochs, lr, curvature_lambda=CURVATURE_LAMBDA):
    """Аналог train_nrc_strong (импортирован выше как есть) — тот же
    build_input/loss/optimizer-протокол, но модель — NRCKANBaseline
    (models/kan_baseline.py), и к NRCRelativeL2Loss добавлен curvature-
    penalty с фиксированным curvature_lambda (TZ_stage8 §3.4)."""
    obs = batch["obs"].to(device)
    cold = batch["cold"].to(device)
    conf = batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    obs_dim = obs.shape[-1]

    spatial = get_spatial(batch, device)
    model = NRCKANBaseline(
        obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
        staleness_dim=4, spatial_dim=SPATIAL_DIM,
        n_hidden_layers=KAN_N_HIDDEN_LAYERS,
        grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER,
    ).to(device)
    u = NRCKANBaseline.build_input(obs, cold, conf, use_staleness=True,
                                    extra_staleness=[disocc, mismatch], spatial=spatial)
    true = batch["true"].to(device)
    loss_fn = NRCRelativeL2Loss(use_confidence_weight=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loss_last, curv_loss_last = None, None
    for ep in range(epochs):
        optimizer.zero_grad()
        pred = model(u)
        data_loss = loss_fn(pred, true)
        curv_loss = model.curvature_regularization_loss()
        loss = data_loss + curvature_lambda * curv_loss
        loss.backward()
        optimizer.step()
        if ep == epochs - 1:
            data_loss_last, curv_loss_last = float(data_loss.item()), float(curv_loss.item())
    # диагностика TZ_stage8 §3.4 — печатается в run_sweep, не хранится в results.json
    model._last_data_loss = data_loss_last
    model._last_curv_loss = curv_loss_last
    return model


def predict_nrc_kan(model, batch, device):
    obs = batch["obs"].to(device)
    cold, conf = batch["cold"].to(device), batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    with torch.no_grad():
        spatial = get_spatial(batch, device)
        u = NRCKANBaseline.build_input(obs, cold, conf, use_staleness=True,
                                        extra_staleness=[disocc, mismatch], spatial=spatial)
        return model(u).cpu().numpy()
