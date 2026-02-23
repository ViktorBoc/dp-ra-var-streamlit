from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .ra import compute_ra
from .stresses import ShockEngine


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: str = ""


def check_discount_factors_monotone(discount_factors: np.ndarray) -> CheckResult:
    ok = bool(np.all(discount_factors[:-1] >= discount_factors[1:]))
    return CheckResult(
        name="Diskontný faktor neklesá v čase",
        passed=ok,
        details="" if ok else "Bol nájdený nárast diskontného faktora.",
    )

def check_ra_monotone(ra_by_p: List[Tuple[float, float]], tol: float = 1e-9) -> CheckResult:
    ps = [p for p, _ in ra_by_p]
    ras = [ra for _, ra in ra_by_p]
    diffs = np.diff(ras)
    ok = bool(np.all(diffs >= -tol))
    if ok:
        return CheckResult(name="RA neklesá s percentilom", passed=True)
    i = int(np.where(diffs < -tol)[0][0])
    return CheckResult(
        name="RA neklesá s percentilom",
        passed=False,
        details=f"RA kleslo z p={ps[i]:.3f} to p={ps[i+1]:.3f} ({ras[i]:.6g} -> {ras[i+1]:.6g}).",
    )


def check_stressed_probs_in_range(
    qx_min: float, qx_max: float, lapse_min: float, lapse_max: float
) -> List[CheckResult]:
    qx_ok = (qx_min >= 0.0) and (qx_max <= 1.0)
    lapse_ok = (lapse_min >= 0.0) and (lapse_max <= 1.0)
    return [
        CheckResult(
            name="qx po strese v [0,1]",
            passed=qx_ok,
            details=f"min={qx_min:.6g}, max={qx_max:.6g}",
        ),
        CheckResult(
            name="lapse po strese v [0,1]",
            passed=lapse_ok,
            details=f"min={lapse_min:.6g}, max={lapse_max:.6g}",
        ),
    ]


def run_consistency_checks(
    insurance_type: str,
    policies,
    assumptions,
    corr: pd.DataFrame,
    product_components: List[str],
    shock_engine: ShockEngine,
    var_levels: List[float],
    index_base: np.ndarray,
    index_stressed: np.ndarray,
) -> List[CheckResult]:
    checks: List[CheckResult] = []

    checks.append(check_discount_factors_monotone(assumptions.discount_factors))

    ra_by_p = []
    for p in var_levels:
        r = compute_ra(
            insurance_type=insurance_type,
            percentile=float(p),
            policies=policies,
            assumptions=assumptions,
            corr=corr,
            product_components=product_components,
            shock_engine=shock_engine,
            index_base=index_base,
            index_stressed=index_stressed,
        )
        ra_by_p.append((float(p), float(r.ra_total)))
    checks.append(check_ra_monotone(ra_by_p))

    p_max = float(max(var_levels))

    # qx bounds (mortality/longevity)
    qx_samples = []
    for age in range(0, 106):
        base = assumptions.mortality_qx_by_age.get(age)
        if base is None:
            continue
        if "mortality" in product_components:
            qx_samples.append(float(base) * float(shock_engine.scaled_value("mortality", p_max)))
        if "longevity" in product_components:
            qx_samples.append(float(base) * float(shock_engine.scaled_value("longevity", p_max)))
    if qx_samples:
        qx_arr = np.clip(np.array(qx_samples, dtype=float), 0.0, 1.0)
        qx_min = float(np.min(qx_arr))
        qx_max = float(np.max(qx_arr))
    else:
        qx_min = 0.0
        qx_max = 0.0

    # lapse bounds (lapse shocks)
    if "lapse" in product_components:
        base_curve = assumptions.lapse_by_product_duration.get(insurance_type, np.zeros(50))
        lapses = []
        for risk in ("lapse_up", "lapse_down", "mass_lapse"):
            lapses.append(shock_engine.apply_lapse(base_curve, risk, p_max))
        lapse_arr = np.concatenate(lapses) if lapses else np.array([0.0])
        lapse_min = float(np.min(lapse_arr))
        lapse_max = float(np.max(lapse_arr))
    else:
        lapse_min = 0.0
        lapse_max = 0.0

    checks.extend(check_stressed_probs_in_range(qx_min, qx_max, lapse_min, lapse_max))
    return checks
