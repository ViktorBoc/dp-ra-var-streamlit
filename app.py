from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from src.io import load_all_inputs
from src.cashflows import Assumptions
from src.stresses import ShockEngine
from src.ra import compute_ra
from src.validation import run_consistency_checks
from src.utils import (
    Policy,
    VALUATION_DATE,
    to_date,
    current_age,
    duration_years,
    remaining_term_years,
    projection_horizon_years,
)

# ---------- Helpers ----------

DEFAULT_PREMIUM_RATE_IF_MISSING = 0.02  # 2% of sum_insured (MVP default)


def _standardize_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize premium column name
    if "premium" not in df.columns:
        if "annual_premium" in df.columns:
            df.rename(columns={"annual_premium": "premium"}, inplace=True)
    if "premium" not in df.columns:
        df["premium"] = np.nan

    needed = [
        "insurance_type",
        "agreement_state",
        "date_of_birth",
        "issue_date",
        "insurance_term",
        "sum_insured",
        "premium",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"dummy_data.csv is missing columns: {missing}")

    return df[needed]


def _build_policies(portfolio_df: pd.DataFrame) -> List[Policy]:
    policies: List[Policy] = []
    for i, r in enumerate(portfolio_df.itertuples(index=False)):
        dob = to_date(r.date_of_birth)
        iss = to_date(r.issue_date)
        iterm = int(r.insurance_term)
        si = float(r.sum_insured)

        prem = getattr(r, "premium", np.nan)
        if prem is None or (isinstance(prem, float) and np.isnan(prem)):
            prem = DEFAULT_PREMIUM_RATE_IF_MISSING * si
        prem = float(prem)

        age = current_age(dob, VALUATION_DATE)
        dur = duration_years(iss, VALUATION_DATE)
        rem = remaining_term_years(iterm, dur)
        horiz = projection_horizon_years(age, rem)

        policies.append(
            Policy(
                policy_id=int(i),
                insurance_type=str(r.insurance_type),
                agreement_state=str(r.agreement_state),
                date_of_birth=dob,
                issue_date=iss,
                insurance_term=iterm,
                sum_insured=si,
                premium=prem,
                duration=dur,
                current_age=age,
                remaining_term=rem,
                horizon=horiz,
            )
        )
    return policies



def _build_assumptions(raw: Dict) -> Assumptions:
    mort = raw["mortality"].copy()
    # mortality.csv may have Slovak column name 'vek'
    if "age" not in mort.columns and "vek" in mort.columns:
        mort.rename(columns={"vek": "age"}, inplace=True)
    if "age" not in mort.columns or "qx" not in mort.columns:
        raise ValueError("mortality.csv must have columns: age (or vek), qx")

    mort_map = {int(a): float(q) for a, q in zip(mort["age"], mort["qx"])}

    lapse = raw["lapse"]
    lapse_by_prod: Dict[str, np.ndarray] = {}
    for prod, g in lapse.groupby("product_type"):
        arr = np.zeros(50, dtype=float)
        for _, r in g.iterrows():
            d = int(r["duration_year"])
            if 1 <= d <= 50:
                arr[d - 1] = float(r["lapse_rate"])
        # forward-fill zeros if any gaps
        for k in range(50):
            if k > 0 and arr[k] == 0.0:
                arr[k] = arr[k - 1]
        lapse_by_prod[str(prod)] = arr

    rf = raw["risk_free"].copy()
    rf = rf.sort_values("year")
    df = rf["discount_factor"].to_numpy(dtype=float)
    if len(df) < 50:
        df = np.pad(df, (0, 50 - len(df)), mode="edge")
    df = df[:50]

    infl = raw["inflation"].copy().sort_values("year")
    idx_base = infl["index_base"].to_numpy(dtype=float)
    idx_st = infl["index_stressed"].to_numpy(dtype=float)
    if len(idx_base) < 50:
        idx_base = np.pad(idx_base, (0, 50 - len(idx_base)), mode="edge")
        idx_st = np.pad(idx_st, (0, 50 - len(idx_st)), mode="edge")
    idx_base = idx_base[:50]
    idx_st = idx_st[:50]

    exp = raw["expenses"].get("assumptions", raw["expenses"])
    return Assumptions(
        mortality_qx_by_age=mort_map,
        lapse_by_product_duration=lapse_by_prod,
        discount_factors=df,
        index_base=idx_base,
        index_stressed=idx_st,
        expenses=exp,
    )


def _ensure_outputs_dir() -> Path:
    root = Path(__file__).resolve().parent
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@st.cache_data(show_spinner=False)
def _load_and_prepare():
    raw = load_all_inputs()

    port = _standardize_portfolio(raw["portfolio"])
    # filter active agreement_state only
    port = port[port["agreement_state"].isin(["new", "paid_up"])].copy()

    assumptions = _build_assumptions(raw)
    shock_engine = ShockEngine(raw["risk_shocks"])
    corr = raw["corr"]
    prm = raw["product_risk_map"]["products"]
    var_levels = raw["var_levels"]
    return port, assumptions, shock_engine, corr, prm, var_levels


# ---------- UI ----------

st.set_page_config(page_title="RA (VaR) – Life Insurance", layout="wide")
st.title("Risk Adjustment (VaR) – Life Insurance")

port, assumptions, shock_engine, corr, product_map, var_levels = _load_and_prepare()
insurance_types = sorted(port["insurance_type"].unique().tolist())

def _invalidate():
    st.session_state["do_compute"] = False

with st.sidebar:
    st.header("Inputs")
    sel_type = st.selectbox(
        "insurance_type",
        insurance_types,
        key="insurance_type",
        on_change=_invalidate,
    )
    sel_p = st.selectbox(
        "percentile p",
        [float(x) for x in var_levels],
        index=len(var_levels) - 1,
        key="percentile",
        on_change=_invalidate,
    )
    compute_btn = st.button("Compute", type="primary")


# Compute only on first load or after explicit click (avoids recompute-heavy checks on each widget change)
if "do_compute" not in st.session_state:
    st.session_state["do_compute"] = False
if compute_btn:
    st.session_state["do_compute"] = True

if st.session_state["do_compute"]:
    port_sel = port[port["insurance_type"] == sel_type].copy()
    policies = _build_policies(port_sel)

    components = product_map.get(sel_type, {}).get(
        "components", ["mortality", "longevity", "lapse", "expense"]
    )
    ra = compute_ra(
        insurance_type=sel_type,
        percentile=sel_p,
        policies=policies,
        assumptions=assumptions,
        corr=corr,
        product_components=components,
        shock_engine=shock_engine,
        index_base=assumptions.index_base,
        index_stressed=assumptions.index_stressed,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("RA_total (EUR)", f"{ra.ra_total:,.2f}")
    c2.metric("RA_rate (%)", f"{(ra.ra_rate*100):.4f}")
    c3.metric("BEL_base (EUR)", f"{ra.bel_base:,.2f}")

    st.subheader("SCR components (portfolio)")
    scr_df = pd.DataFrame(
        [{"component": k, "SCR": float(v)} for k, v in ra.scr_components.items()]
    )
    scr_total = float(ra.ra_total)
    scr_df = pd.concat(
        [scr_df, pd.DataFrame([{"component": "TOTAL (aggregated)", "SCR": scr_total}])],
        ignore_index=True,
    )
    st.dataframe(
        scr_df.style.format({"SCR": "{:,.2f}"}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Consistency checks")
    checks = run_consistency_checks(
        insurance_type=sel_type,
        policies=policies,
        assumptions=assumptions,
        corr=corr,
        product_components=components,
        shock_engine=shock_engine,
        var_levels=[float(x) for x in var_levels],
        index_base=assumptions.index_base,
        index_stressed=assumptions.index_stressed,
    )
    chk_df = pd.DataFrame(
        [{"check": c.name, "status": "PASS" if c.passed else "FAIL", "details": c.details} for c in checks]
    )
    st.dataframe(chk_df, width="stretch", hide_index=True)

    st.subheader("Export")

    # priprav export DF
    ra_export_df = pd.DataFrame([{
        "insurance_type": ra.insurance_type,
        "percentile": ra.percentile,
        "BEL_base": ra.bel_base,
        "RA_total": ra.ra_total,
        "RA_rate": ra.ra_rate,
    }])

    scr_export_df = pd.DataFrame([{"component": k, "SCR": v} for k, v in ra.scr_components.items()])

    c_dl, c_save = st.columns(2)

    with c_dl:
        st.markdown("**Download (recommended for deployed app)**")
        st.download_button(
            label="Download ra_results.csv",
            data=ra_export_df.to_csv(index=False).encode("utf-8"),
            file_name="ra_results.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download scr_components.csv",
            data=scr_export_df.to_csv(index=False).encode("utf-8"),
            file_name="scr_components.csv",
            mime="text/csv",
        )

    with c_save:
        st.markdown("**Save to outputs/ (local / debug)**")
        out_dir = _ensure_outputs_dir()
        if st.button("Save CSV to outputs/"):
            ra_export_df.to_csv(out_dir / "ra_results.csv", index=False, encoding="utf-8")
            scr_export_df.to_csv(out_dir / "scr_components.csv", index=False, encoding="utf-8")
            st.success(f"Saved to {out_dir}")
        st.caption("On Streamlit Cloud this saves on the server (not to the user's PC). Use Download.")

