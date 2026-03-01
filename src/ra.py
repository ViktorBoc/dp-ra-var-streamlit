from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .scr import compute_scr_components, SCRResult
from .stresses import ShockEngine
from .utils import safe_div

@dataclass(frozen=True)
class RAResult:
    insurance_type: str
    percentile: float
    bel_base: float
    ra_total: float
    ra_rate: float
    scr_components: Dict[str, float]
    bel_stressed_by_component: Dict[str, float]

def compute_ra(
    insurance_type: str,
    percentile: float,
    policies,
    assumptions,
    corr: pd.DataFrame,
    product_components: List[str],
    shock_engine: ShockEngine,
    index_base,
    index_stressed,
) -> RAResult:
    scr_res: SCRResult = compute_scr_components(
        policies=policies,
        a=assumptions,
        shock_engine=shock_engine,
        corr=corr,
        product_components=product_components,
        p=float(percentile),
        index_base=index_base,
        index_stressed=index_stressed,
    )
    ra_total = float(scr_res.scr_total)
    bel = float(scr_res.bel_base)
    ra_rate = safe_div(ra_total, bel, default=0.0)
    return RAResult(
        insurance_type=insurance_type,
        percentile=float(percentile),
        bel_base=bel,
        ra_total=ra_total,
        ra_rate=ra_rate,
        scr_components=scr_res.scr_by_component,
        bel_stressed_by_component=scr_res.bel_stressed_by_component,
    )
