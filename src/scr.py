from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .cashflows import Assumptions, Scenario, portfolio_bel
from .stresses import ShockEngine

COMPONENTS = ["mortality", "longevity", "lapse", "expense"]

@dataclass(frozen=True)
class SCRResult:
    bel_base: float
    scr_by_component: Dict[str, float]
    bel_stressed_by_component: Dict[str, float]
    scr_total: float

def aggregate_scr(scr_by_component: Dict[str, float], corr: pd.DataFrame) -> float:
    order = list(corr.columns)
    v = np.array([float(scr_by_component.get(k, 0.0)) for k in order], dtype=float)
    C = corr.loc[order, order].values.astype(float)
    x = float(v.T @ C @ v)
    return float(np.sqrt(max(0.0, x)))

def compute_scr_components(
    policies,
    a: Assumptions,
    shock_engine: ShockEngine,
    corr: pd.DataFrame,
    product_components: List[str],
    p: float,
    index_base: np.ndarray,
    index_stressed: np.ndarray,
) -> SCRResult:
    base = portfolio_bel(policies, a, Scenario())
    bel_base = float(base["bel"])

    scr = {k: 0.0 for k in COMPONENTS}
    bel_stressed = {k: bel_base for k in COMPONENTS}

    # Mortality
    if "mortality" in product_components:
        qx_mult = shock_engine.scaled_value("mortality", p)
        bel = portfolio_bel(policies, a, Scenario(qx_multiplier=qx_mult))["bel"]
        bel_stressed["mortality"] = float(bel)
        scr["mortality"] = max(0.0, float(bel) - bel_base)

    # Longevity
    if "longevity" in product_components:
        qx_mult = shock_engine.scaled_value("longevity", p)
        bel = portfolio_bel(policies, a, Scenario(qx_multiplier=qx_mult))["bel"]
        bel_stressed["longevity"] = float(bel)
        scr["longevity"] = max(0.0, float(bel) - bel_base)

    # Lapse
    if "lapse" in product_components:
        def portfolio_bel_with_lapse(risk_name: str) -> float:
            total_bel = 0.0
            for pol in policies:
                T = pol.horizon
                if T <= 0:
                    continue
                base_curve = a.lapse_by_product_duration.get(pol.insurance_type, np.zeros(50))
                lapse_base = np.array(
                    [base_curve[min(49, max(0, (pol.duration + t) - 1))] for t in range(1, T + 1)],
                    dtype=float,
                )
                lapse_st = shock_engine.apply_lapse(lapse_base, risk_name, p)
                total_bel += portfolio_bel([pol], a, Scenario(lapse_rates=lapse_st))["bel"]
            return float(total_bel)

        bel_up = portfolio_bel_with_lapse("lapse_up")
        bel_down = portfolio_bel_with_lapse("lapse_down")
        bel_mass = portfolio_bel_with_lapse("mass_lapse")

        scr_up = max(0.0, bel_up - bel_base)
        scr_down = max(0.0, bel_down - bel_base)
        scr_mass = max(0.0, bel_mass - bel_base)

        # worst case scenario
        worst = max(scr_up, scr_down, scr_mass)
        if worst == scr_up:
            bel_stressed["lapse"] = bel_up
        elif worst == scr_down:
            bel_stressed["lapse"] = bel_down
        else:
            bel_stressed["lapse"] = bel_mass
        scr["lapse"] = worst

    # Expense
    if "expense" in product_components:
        level = shock_engine.scaled_value("expense_level", p)
        scale = shock_engine.scale_factor(p)
        infl_index = shock_engine.blend_inflation_index(index_base, index_stressed, scale)
        bel = portfolio_bel(
            policies, a, Scenario(expense_level_multiplier=level, inflation_index=infl_index)
        )["bel"]
        bel_stressed["expense"] = float(bel)
        scr["expense"] = max(0.0, float(bel) - bel_base)

    total = aggregate_scr(scr, corr)
    return SCRResult(
        bel_base=bel_base,
        scr_by_component=scr,
        bel_stressed_by_component=bel_stressed,
        scr_total=float(total),
    )
