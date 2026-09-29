"""Обучение и инференс nrc_strong (patent-faithful MLP) — MLP-сторона сравнения с KAN.

Вынесено без изменений логики из run_capacity_sweep_zeroday_strong_nrc.py (TZ_stage6).
"""
import torch
from gi_surrogate.data.batch import SPATIAL_DIM, get_spatial
from gi_surrogate.models.losses import NRCRelativeL2Loss
from gi_surrogate.models.nrc import NRCStyleBaseline

STRONG_N_HIDDEN_LAYERS = 5   # TZ_stage6 §2.1, патент-верный конфиг

STRONG_BIAS = False          # TZ_stage6 §2.1, патент-верный конфиг

NRC_DEVICE = torch.device("cpu")  # исправление §9: устраняет MPS-джиттер для NRC


def train_nrc_strong(hidden_dim, batch, device, epochs, lr):
    """Патент-верный конфиг (5 слоёв, bias=False), по образцу _train_nrc_boosted
    из run_nrc_boosted_zeroday.py, но с переменным hidden_dim вместо
    зашитого BOOSTED_HIDDEN_DIM=64 — впервые sweep по ширине для этого конфига."""
    obs = batch["obs"].to(device)
    cold = batch["cold"].to(device)
    conf = batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    obs_dim = obs.shape[-1]

    spatial = get_spatial(batch, device)
    model = NRCStyleBaseline(
        obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
        staleness_dim=4, spatial_dim=SPATIAL_DIM,
        n_hidden_layers=STRONG_N_HIDDEN_LAYERS, bias=STRONG_BIAS,
    ).to(device)
    u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                      extra_staleness=[disocc, mismatch], spatial=spatial)
    true = batch["true"].to(device)
    loss_fn = NRCRelativeL2Loss(use_confidence_weight=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(u)
        loss = loss_fn(pred, true)
        loss.backward()
        optimizer.step()
    return model


def predict_nrc_strong(model, batch, device):
    obs = batch["obs"].to(device)
    cold, conf = batch["cold"].to(device), batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    with torch.no_grad():
        spatial = get_spatial(batch, device)
        u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                          extra_staleness=[disocc, mismatch], spatial=spatial)
        return model(u).cpu().numpy()
