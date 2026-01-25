from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import floor

import numpy as np

VALUATION_DATE = date(2026, 1, 1)
MAX_AGE = 105
MAX_HORIZON_YEARS = 50


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def safe_div(n: float, d: float, default: float = 0.0, eps: float = 1e-12) -> float:
    return default if abs(d) < eps else n / d


def to_date(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError("Missing date")
    return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()


def year_diff_floor(d1: date, d0: date) -> int:
    """Floor of year fraction using 365.25 day convention (as requested)."""
    return floor((d1 - d0).days / 365.25)


def current_age(dob: date, valuation_date: date = VALUATION_DATE) -> int:
    return year_diff_floor(valuation_date, dob)


def duration_years(issue_date: date, valuation_date: date = VALUATION_DATE) -> int:
    return year_diff_floor(valuation_date, issue_date)


def remaining_term_years(insurance_term: int, duration: int) -> int:
    if insurance_term == 9999:
        return 9999
    return max(0, int(insurance_term) - int(duration))


def projection_horizon_years(current_age_years: int, remaining_term: int) -> int:
    cap_age = max(0, MAX_AGE - int(current_age_years))
    cap_term = remaining_term if remaining_term != 9999 else 9999
    return int(min(cap_term, MAX_HORIZON_YEARS, cap_age))


@dataclass(frozen=True)
class Policy:
    policy_id: int
    insurance_type: str
    agreement_state: str
    date_of_birth: date
    issue_date: date
    insurance_term: int
    sum_insured: float
    premium: float  # annual premium; may be 0
    duration: int
    current_age: int
    remaining_term: int
    horizon: int
