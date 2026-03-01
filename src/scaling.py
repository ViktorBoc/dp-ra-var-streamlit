from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import norm

@dataclass(frozen=True)
class VarScaler:
    base_confidence: float = 0.995

    def z(self, p: float) -> float:
        return float(norm.ppf(p))

    def scale(self, p: float) -> float:
        z_base = self.z(self.base_confidence)
        z_p = self.z(p)
        return 0.0 if z_base == 0.0 else float(z_p / z_base)

    def scale_multiplier(self, shock_0995: float, p: float) -> float:
        s = self.scale(p)
        return 1.0 + (float(shock_0995) - 1.0) * s

    def scale_additive(self, delta_0995: float, p: float) -> float:
        s = self.scale(p)
        return float(delta_0995) * s
