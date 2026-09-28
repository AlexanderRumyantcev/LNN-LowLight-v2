"""NRC-style baseline (patent-faithful MLP) — MLP-сторона сравнения с KAN.

Перенесён без изменений из models/baselines.py (класс NRCStyleBaseline). Остальные
baseline'ы (NRD, GRU, ODE-LSTM) не нужны активным тестам и остались в архиве.
"""
import torch
import torch.nn as nn


class NRCStyleBaseline(nn.Module):
    """
    Online per-scene per-frame MLP (аналог NVIDIA NRC / AMD FSR Radiance Cache) — БЕЗ
    рекуррентности: каждый шаг обрабатывается независимо тем же MLP (как в оригинальном
    NRC, где каждый query — независимый forward-pass, а обучение online происходит по
    накопленной статистике сцены, а не через carried-over hidden state между кадрами).

    faithful (use_staleness=False): вход = только obs, как в оригинальном NRC.
    честная версия (use_staleness=True): вход = obs + [cold_start, confidence], как у CfC-B
        (§2.2) — тот же staleness-вектор, тот же источник (log1p(spp) нормализованный).

    Dense-путь (§4.2 TZ_stage1b): staleness_dim расширяется до 4
        ([cold_start, confidence, disocclusion_flag, geometric_mismatch_score]) —
        тот же принцип, что CfCProbeModule.staleness_dim, для симметрии сравнения.
        По умолчанию staleness_dim=2 (probe-путь, поведение без изменений).
    """

    def __init__(self, obs_dim: int = 1, hidden_dim: int = 32, use_staleness: bool = False,
                 staleness_dim: int = 2, spatial_dim: int = 0, n_hidden_layers: int = 2,
                 bias: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.use_staleness = use_staleness
        self.staleness_dim = staleness_dim
        # bias (09.09.2026, TZ_stage6_nrc_patent_capacity_check.md §3.2): True по умолчанию =
        # старое поведение (обратная совместимость со всеми существующими 14 конфигурациями
        # MODEL_KINDS). False — patent-faithful режим (US11610360, MLP без bias во всех слоях),
        # используется ТОЛЬКО новым kind'ом nrc_honest_boosted (run_nrc_boosted_zeroday.py).
        self.bias = bias
        # spatial_dim (2026-08-20, models/spatial_features.py): позиционное кондиционирование
        # (позиция+направление+normal+albedo, positional encoding) — закрывает найденный разрыв
        # с настоящим NRC (Müller et al. 2021), см. чат/mempalace 2026-08-19/20. 0 по умолчанию
        # = старое поведение без изменений (обратная совместимость).
        self.spatial_dim = spatial_dim
        # n_hidden_layers (2026-08-20, чат — литературная проверка после эксперимента с
        # деградацией nrc_honest_spatial): 2 по умолчанию = старое поведение (обратная
        # совместимость). Референс из литературы (Neural Radiance Cache Implementation on
        # Mobile GPU, SIGGRAPH Asia 2025, сравнение с оригинальным Müller et al. 2021;
        # tiny-cuda-nn DOCUMENTATION.md конфиг, воспроизводящий encoding NRC) — настоящий NRC
        # использует 5 скрытых слоёв по 64 нейрона, у нас было 2×32. Neural Visibility Cache
        # for Real-Time Light Sampling (arXiv 2506.05930) прямо пишет: NRC требует БОЛЬШЕЙ MLP
        # и БОЛЬШЕГО числа шагов обучения для приемлемых результатов, чем более простые
        # альтернативы — литературное подтверждение гипотезы недообучения после расширения
        # входа spatial-кондиционированием (48 доп. измерений).
        self.n_hidden_layers = n_hidden_layers
        input_dim = obs_dim + (staleness_dim if use_staleness else 0) + spatial_dim
        layers = [nn.Linear(input_dim, hidden_dim, bias=bias), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, obs_dim, bias=bias))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def build_input(
        obs: torch.Tensor,
        cold_start: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        use_staleness: bool = False,
        extra_staleness: list[torch.Tensor] | None = None,
        spatial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Тот же формат входа, что CfCProbeModule.build_input (§3.1/§3.2) — для честного
        сравнения оба baseline'а и CfC-B видят идентично собранный staleness-вектор.
        extra_staleness — доп. каналы (§4.2 dense-путь: [disocclusion_flag,
        geometric_mismatch_score]), None по умолчанию = старое 2-элементное поведение.
        spatial — [B, T, spatial_dim] позиционное кондиционирование (models/spatial_
        features.py::build_spatial_conditioning), None по умолчанию = старое поведение без
        пространственного входа. Порядок каналов: obs, затем staleness (если есть), затем
        spatial (если есть) — должен совпадать с spatial_dim, переданным модели в __init__."""
        parts = [obs]
        if use_staleness:
            if cold_start is None or confidence is None:
                raise ValueError("use_staleness=True требует cold_start и confidence")
            components = [cold_start, confidence]
            if extra_staleness:
                components.extend(extra_staleness)
            parts.append(torch.stack(components, dim=-1))
        if spatial is not None:
            parts.append(spatial)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def forward(self, u_seq: torch.Tensor) -> torch.Tensor:
        """
        u_seq: [B, T, input_dim] (см. build_input)
        Returns: pred_seq [B, T, obs_dim] — каждый шаг обработан НЕЗАВИСИМО, без
                 рекуррентности (архитектурное отличие от CfC-B/GRU-baseline'ов).
        """
        return self.mlp(u_seq)
