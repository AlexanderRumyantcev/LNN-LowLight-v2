"""
models/kan_baseline.py — TZ_stage8 (13.09.2026): NRC-style baseline с KAN
(efficient-kan, Blealtan/efficient-kan) вместо MLP внутри NRCStyleBaseline.

Интерфейс идентичен NRCStyleBaseline (models/baselines.py) — тот же
build_input (переиспользуется как staticmethod), тот же forward-контракт
(per-frame, без рекуррентности) — для прямой подстановки в существующие
capacity sweep скрипты без переписывания train/predict-обвязки.

Отличие: self.mlp (nn.Sequential[Linear,ReLU,...]) заменён на стек
KANLinear-слоёв (efficient_kan.KANLinear). grid_size/spline_order
ФИКСИРОВАНЫ на дефолтах efficient-kan (5/3) — TZ_stage8 §3.2, не
тюнятся в этом этапе.

Curvature-регуляризация (TZ_stage8 §3.4): штраф на вторую разность
control points каждого KANLinear-слоя (spline_weight, форма
[out_features, in_features, grid_size+spline_order]) — дискретный
аналог penalty на вторую производную сплайна (arXiv 2411.06727),
адресует задокументированную чувствительность KAN к шуму (arXiv
2407.14882) на MC-шумном Zero Day таргете. НЕ путать с
KANLinear.regularization_loss (встроенная в efficient-kan L1+entropy
регуляризация из оригинальной статьи KAN — другая цель, разреженность/
интерпретируемость, не сглаживание; в этом этапе не используется).
"""
import torch
import torch.nn as nn
from efficient_kan import KANLinear

from gi_surrogate.models.nrc import NRCStyleBaseline


class NRCKANBaseline(nn.Module):
    """KAN-версия NRCStyleBaseline. См. models/baselines.py::NRCStyleBaseline
    докстринг для контекста faithful/честной схемы staleness-входа — здесь
    та же схема, специфика TZ_stage8 только в самом регрессоре."""

    def __init__(self, obs_dim: int = 1, hidden_dim: int = 32, use_staleness: bool = False,
                 staleness_dim: int = 2, spatial_dim: int = 0, n_hidden_layers: int = 5,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.use_staleness = use_staleness
        self.staleness_dim = staleness_dim
        self.spatial_dim = spatial_dim
        self.n_hidden_layers = n_hidden_layers
        self.grid_size = grid_size
        self.spline_order = spline_order

        input_dim = obs_dim + (staleness_dim if use_staleness else 0) + spatial_dim
        # layers_hidden: input_dim -> hidden_dim (x n_hidden_layers) -> obs_dim —
        # структурное соответствие NRCStyleBaseline: n_hidden_layers слоёв
        # ширины hidden_dim между входом и выходом (TZ_stage8 §3.2)
        dims = [input_dim] + [hidden_dim] * n_hidden_layers + [obs_dim]
        self.kan_layers = nn.ModuleList([
            KANLinear(dims[i], dims[i + 1], grid_size=grid_size, spline_order=spline_order)
            for i in range(len(dims) - 1)
        ])

    # build_input идентичен NRCStyleBaseline (TZ_stage8 §3.2 — та же схема
    # staleness/spatial-входа, для честного сравнения на одном eval-батче)
    build_input = staticmethod(NRCStyleBaseline.build_input)

    def forward(self, u_seq: torch.Tensor) -> torch.Tensor:
        """u_seq: [B, T, input_dim] (см. build_input). Каждый шаг обработан
        независимо (без рекуррентности, как у NRCStyleBaseline) — KANLinear
        принимает произвольную форму входа с последней осью == in_features
        (efficient_kan.KANLinear.forward делает x.reshape(-1, in_features) и
        reshape обратно), поэтому [B, T, D] прогоняется целиком без ручного
        цикла по T."""
        x = u_seq
        for layer in self.kan_layers:
            x = layer(x)
        return x

    def curvature_regularization_loss(self) -> torch.Tensor:
        """TZ_stage8 §3.4 — сумма по слоям среднего квадрата второй разности
        control points (spline_weight, ось -1 = координаты control points
        одного univariate-сплайна, grid_size+spline_order штук) каждого
        KANLinear. Вторая разность — дискретная аппроксимация второй
        производной сплайна, тот же смысловой penalty, что в arXiv
        2411.06727 (\\int (d^2 S/dx^2)^2 dx), только в терминах B-spline
        коэффициентов вместо самой функции — не требует интегрирования,
        дёшево считается прямо по параметрам модели."""
        total = torch.zeros((), device=self.kan_layers[0].spline_weight.device)
        for layer in self.kan_layers:
            w = layer.spline_weight  # [out_features, in_features, n_coef]
            d2 = w[..., 2:] - 2 * w[..., 1:-1] + w[..., :-2]
            total = total + (d2 ** 2).mean()
        return total
