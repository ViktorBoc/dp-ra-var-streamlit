from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .scaling import VarScaler
from .utils import clamp

@dataclass(frozen=True)
class ShockDefinitions:
    risks: Dict[str, dict]
    base_confidence: float = 0.995

class ShockEngine:
    """Scale shocks from 99.5% to arbitrary p using normal quantile ratio."""

    def __init__(self, risk_shocks_json: dict):
        self._raw = risk_shocks_json
        base_conf = float(risk_shocks_json.get("base_confidence", 0.995))
        self.scaler = VarScaler(base_confidence=base_conf)
        self.risks = risk_shocks_json["risks"]

    def scale_factor(self, p: float) -> float:
        return self.scaler.scale(p)

    def scaled_value(self, risk_name: str, p: float) -> float:
        d = self.risks[risk_name]
        t = d["type"]
        shock_0995 = float(d["shock_0.995"])
        if t in ("multiplier", "multiplier_with_absolute_cap"):
            return self.scaler.scale_multiplier(shock_0995, p)
        if t in ("additive", "instant_proportion"):
            return self.scaler.scale_additive(shock_0995, p)
        raise ValueError(f"Unknown shock type: {t}")

    def apply_qx_multiplier(self, qx: np.ndarray, risk_name: str, p: float) -> np.ndarray:
        mult = self.scaled_value(risk_name, p)
        q = qx * mult
        return np.clip(q, 0.0, 1.0)

    def apply_lapse(self, lapse: np.ndarray, risk_name: str, p: float) -> np.ndarray:
        d = self.risks[risk_name]
        t = d["type"]
        if risk_name == "mass_lapse" or t == "instant_proportion":
            rate = self.scaled_value(risk_name, p)
            out = lapse.copy()
            if len(out) > 0:
                out[0] = max(out[0], clamp(rate, 0.0, 1.0))
            return np.clip(out, 0.0, 1.0)

        if t == "multiplier":
            mult = self.scaled_value(risk_name, p)
            return np.clip(lapse * mult, 0.0, 1.0)

        if t == "multiplier_with_absolute_cap":
            mult = self.scaled_value(risk_name, p)
            abs_cap = float(d.get("absolute_decrease_cap", 0.0))
            stressed = np.maximum(lapse * mult, lapse - abs_cap)
            return np.clip(stressed, 0.0, 1.0)

        raise ValueError(f"Unsupported lapse shock: {risk_name} / {t}")

    @staticmethod
    def blend_inflation_index(index_base: np.ndarray, index_stressed: np.ndarray, scale: float) -> np.ndarray:
        """Geometric blend so that scale=0 => base, scale=1 => stressed."""
        ratio = np.clip(index_stressed / index_base, 1e-30, None)
        return index_base * np.exp(scale * np.log(ratio))
