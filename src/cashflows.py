from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .utils import Policy, clamp


@dataclass(frozen=True)
class Assumptions:
    mortality_qx_by_age: Dict[int, float]            # age -> qx
    lapse_by_product_duration: Dict[str, np.ndarray] # product -> rates[1..50] as array len 50 (index 0==dur 1)
    discount_factors: np.ndarray                     # len 50, year 1..50
    index_base: np.ndarray                           # len 50
    index_stressed: np.ndarray                       # len 50
    expenses: Dict[str, float]                       # base expense params


@dataclass(frozen=True)
class Scenario:
    qx_multiplier: float = 1.0
    lapse_rates: Optional[np.ndarray] = None         # len horizon, if None use base lapse
    expense_level_multiplier: float = 1.0
    inflation_index: Optional[np.ndarray] = None     # len 50 (or >= horizon); if None uses base index


def _qx_for_age(mortality_qx_by_age: Dict[int, float], age: int) -> float:
    if age in mortality_qx_by_age:
        return float(mortality_qx_by_age[age])
    ages = sorted(mortality_qx_by_age.keys())
    if age <= ages[0]:
        return float(mortality_qx_by_age[ages[0]])
    return float(mortality_qx_by_age[ages[-1]])


def _lapse_for_duration(lapse_by_product_duration: Dict[str, np.ndarray], product: str, duration_year: int) -> float:
    arr = lapse_by_product_duration.get(product)
    if arr is None:
        return 0.0
    d = int(duration_year)
    if d <= 1:
        return float(arr[0])
    if d >= 50:
        return float(arr[-1])
    return float(arr[d - 1])


def project_policy_bel(policy: Policy, a: Assumptions, s: Scenario) -> Dict[str, float]:
    """Project yearly cashflows and return PVs and BEL for a single policy."""
    T = int(policy.horizon)
    if T <= 0:
        return {"pv_inflows": 0.0, "pv_outflows": 0.0, "bel": 0.0}

    df = a.discount_factors[:T]
    infl = (s.inflation_index if s.inflation_index is not None else a.index_base)[:T]
    lapse_stressed = s.lapse_rates

    product = policy.insurance_type
    sum_insured = float(policy.sum_insured)
    premium = float(policy.premium)

    # paid_up override
    if str(policy.agreement_state).lower() == "paid_up":
        premium = 0.0

    # expenses
    acq = float(a.expenses.get("acquisition_per_policy", 0.0))
    maint = float(a.expenses.get("maintenance_per_policy", 0.0))
    comm_1 = float(a.expenses.get("commission_first_year_rate", 0.0))
    comm_r = float(a.expenses.get("commission_renewal_rate", 0.0))
    claim_handling = float(a.expenses.get("claim_handling_per_claim", 0.0))

    pv_in = 0.0
    pv_out = 0.0

    S = 1.0  # in-force at start of year
    remaining_term = int(policy.remaining_term)

    for t in range(1, T + 1):
        age = int(policy.current_age + (t - 1))
        qx = _qx_for_age(a.mortality_qx_by_age, age) * float(s.qx_multiplier)
        qx = clamp(qx, 0.0, 1.0)

        dur = int(policy.duration + t)
        lapse_base = _lapse_for_duration(a.lapse_by_product_duration, product, dur)
        if lapse_stressed is None:
            lapse = float(lapse_base)
        else:
            lapse = float(lapse_stressed[t - 1])

        lapse = clamp(lapse, 0.0, 1.0)

        deaths = S * qx
        lapses = (S - deaths) * lapse
        S_next = S - deaths - lapses
        if S_next < 0.0:
            S_next = 0.0

        # inflows
        inflow = 0.0
        if product in ("term_insurance", "whole_of_life", "endowment", "UL_endowment"):
            inflow += premium * S
        elif product == "annuity":
            inflow += 0.0
        else:
            inflow += premium * S

        # outflows: benefits
        outflow = 0.0
        if product in ("term_insurance", "whole_of_life", "endowment", "UL_endowment"):
            outflow += sum_insured * deaths

        if product == "annuity":
            annual_payment = sum_insured
            outflow += annual_payment * S_next

        if product in ("endowment", "UL_endowment"):
            if remaining_term > 0 and t == remaining_term and remaining_term <= T:
                outflow += sum_insured * S_next

        # expenses (outflows)
        idx = float(infl[t - 1])
        exp_mult = float(s.expense_level_multiplier)

        # acquisition: only if duration==0 and agreement_state==new, in first projection year
        if policy.duration == 0 and t == 1 and str(policy.agreement_state).lower() == "new":
            outflow += acq * idx * exp_mult

        outflow += maint * idx * exp_mult * S

        if inflow > 0.0:
            rate = comm_1 if (policy.duration == 0 and t == 1) else comm_r
            outflow += rate * inflow

        if deaths > 0.0:
            outflow += claim_handling * idx * exp_mult * deaths

        pv_in += inflow * float(df[t - 1])
        pv_out += outflow * float(df[t - 1])

        S = S_next
        if S <= 1e-12:
            break

    bel = pv_out - pv_in
    return {"pv_inflows": float(pv_in), "pv_outflows": float(pv_out), "bel": float(bel)}


def portfolio_bel(policies, a: Assumptions, s: Scenario) -> Dict[str, float]:
    pv_in = 0.0
    pv_out = 0.0
    bel = 0.0
    for pol in policies:
        r = project_policy_bel(pol, a, s)
        pv_in += r["pv_inflows"]
        pv_out += r["pv_outflows"]
        bel += r["bel"]
    return {"pv_inflows": float(pv_in), "pv_outflows": float(pv_out), "bel": float(bel)}
