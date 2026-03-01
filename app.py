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

DEFAULT_PREMIUM_RATE_IF_MISSING = 0.02  # 2% of sum_insured

def _standardize_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
        "premium_payment_term",
        "sum_insured",
        "premium",
        "surrender_value",
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

        ppt = int(r.premium_payment_term)
        ppt = min(ppt, iterm)
        remaining_ppt = max(0, ppt - dur)

        sv = float(getattr(r, "surrender_value", 0.0))
        if sv is None or (isinstance(sv, float) and np.isnan(sv)):
            sv = 0.0

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
                premium_payment_term=remaining_ppt,
                surrender_value=sv,
            )
        )
    return policies

def _build_assumptions(raw: Dict) -> Assumptions:
    mort = raw["mortality"].copy()
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

    exp = raw["expenses"]
    return Assumptions(
        mortality_qx_by_age=mort_map,
        lapse_by_product_duration=lapse_by_prod,
        discount_factors=df,
        index_base=idx_base,
        index_stressed=idx_st,
        expenses=exp,
    )

@st.cache_data(show_spinner=False)
def _load_and_prepare():
    raw = load_all_inputs()

    port = _standardize_portfolio(raw["portfolio"])
    port = port[port["agreement_state"].isin(["new", "paid_up"])].copy()

    assumptions = _build_assumptions(raw)
    shock_engine = ShockEngine(raw["risk_shocks"])
    corr = raw["corr"]
    prm = raw["product_risk_map"]["products"]
    var_levels = raw["var_levels"]
    return port, assumptions, shock_engine, corr, prm, var_levels

# ---------- UI ----------
st.set_page_config(page_title="RA (VaR) – Životné poistenie", layout="wide")
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 380px;
        max-width: 380px;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        white-space: normal !important;
        overflow: visible !important;
        height: auto !important;
    }
</style>
""", unsafe_allow_html=True)
st.title("Riziková úprava (VaR) – Životné poistenie")

with st.expander("ℹ️ Vysvetlivky skratiek"):
    st.markdown("""
    - **RA** (Risk Adjustment / Riziková úprava) – prirážka k záväzkom za neistotu v odhadoch budúcich peňažných tokov
    - **BEL** (Best Estimate Liability / Najlepší odhad záväzkov) – súčasná hodnota očakávaných budúcich výplat mínus príjmy z poistného
    - **NFR** (Non-Financial Risk / Nefinančné riziko) – rizikový kapitál pre jednotlivé rizikové komponenty (mortalita, storno, náklady, dlhovekosť)
    - **VaR** (Value at Risk) – štatistická miera rizika na danom percentile spoľahlivosti
    - **FCF** (Fulfillment Cash Flows / Peňažné toky na splnenie záväzkov) – súčet BEL a rizikovej prirážky RA; celkový záväzok poisťovne vykazovaný podľa IFRS 17
    """)

port, assumptions, shock_engine, corr, product_map, var_levels = _load_and_prepare()
EXCLUDED_TYPES = {"UL_endowment"}
insurance_types = sorted([t for t in port["insurance_type"].unique().tolist() if t not in EXCLUDED_TYPES])

def _invalidate():
    st.session_state["do_compute"] = False

with st.sidebar:
    st.header("Vstupy")

    sk_names = {
        "term_insurance": "Rizikové životné poistenie (Term Insurance)",
        "whole_of_life": "Trvalé životné poistenie (Whole of Life)",
        "endowment": "Kapitálové životné poistenie (Endowment)",
        "annuity": "Dôchodkové poistenie (Annuity)",
    }

    sel_type = st.selectbox(
        "Typ poistenia",
        insurance_types,
        key="insurance_type",
        on_change=_invalidate,
        format_func=lambda x: sk_names.get(x, x)
    )
    sel_p = st.selectbox(
        "Percentil p",
        [float(x) for x in var_levels],
        index=len(var_levels) - 1,
        key="percentile",
        on_change=_invalidate,
        format_func=lambda x: f"{x * 100:.1f}%"
    )
    compute_btn = st.button("Vypočítať", type="primary")

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Riziková prirážka RA (EUR)", f"{ra.ra_total:,.2f}")
    c2.metric("Sadzba RA (%)", f"{(ra.ra_rate * 100):.4f}")
    c3.metric("BEL základný (EUR)", f"{ra.bel_base:,.2f}")
    c4.metric("Peňažné toky na splnenie záväzkov - FCF (EUR)", f"{ra.bel_base + ra.ra_total:,.2f}")

    st.subheader("NFR komponenty (portfólio)")
    nfr_names = {
        "mortality": "Mortalita (Mortality)",
        "longevity": "Dlhovekosť (Longevity)",
        "lapse": "Storno (Lapse)",
        "expense": "Náklady (Expense)",
    }
    scr_df = pd.DataFrame([{
        "Komponent": nfr_names.get(k, k),
        "BEL šokovaný (EUR)": ra.bel_stressed_by_component.get(k, ra.bel_base),
        "NFR": float(v),
    } for k, v in ra.scr_components.items()])
    scr_total = float(ra.ra_total)
    scr_df = pd.concat(
        [scr_df, pd.DataFrame([{
            "Komponent": "CELKOM (agregované)",
            "BEL šokovaný (EUR)": "",
            "NFR": scr_total,
        }])],
        ignore_index=True,
    )
    st.dataframe(
        scr_df.style.format({
            "BEL šokovaný (EUR)": lambda x: f"{x:,.2f}" if isinstance(x, float) else x,
            "NFR": "{:,.2f}",
        }),
        width="stretch",
        hide_index=True,
    )

    st.subheader("RA po rokoch (rozpustenie)")


    def _shift_policies_by_year(policies: List[Policy], shift: int) -> List[Policy]:
        shifted = []
        for pol in policies:
            new_horizon = max(0, pol.horizon - shift)
            new_remaining_term = max(0, pol.remaining_term - shift) if pol.remaining_term != 9999 else 9999
            new_age = pol.current_age + shift
            new_duration = pol.duration + shift
            new_ppt = max(0, pol.premium_payment_term - shift)
            shifted.append(Policy(
                policy_id=pol.policy_id,
                insurance_type=pol.insurance_type,
                agreement_state=pol.agreement_state,
                date_of_birth=pol.date_of_birth,
                issue_date=pol.issue_date,
                insurance_term=pol.insurance_term,
                sum_insured=pol.sum_insured,
                premium=pol.premium,
                duration=new_duration,
                current_age=new_age,
                remaining_term=new_remaining_term,
                horizon=new_horizon,
                premium_payment_term=new_ppt,
                surrender_value=pol.surrender_value,
            ))
        return shifted

    ra_rate_locked = ra.ra_rate

    max_horizon = max((p.horizon for p in policies), default=0)
    max_years = min(max_horizon, 50)

    rows = []
    from src.cashflows import portfolio_bel as _pbel, Scenario as _Scenario

    for t in range(max_years + 1):
        if t == 0:
            bel_t = ra.bel_base
        else:
            shifted = _shift_policies_by_year(policies, t)
            bel_t = float(_pbel(shifted, assumptions, _Scenario())["bel"])

        ra_t = ra_rate_locked * bel_t if abs(bel_t) > 1e-6 else 0.0
        rows.append({
            "Rok": t,
            "BEL (EUR)": bel_t,
            "Sadzba RA (%)": ra_rate_locked * 100,
            "RA na začiatku roka (EUR)": ra_t,
        })

    ra_schedule_df = pd.DataFrame(rows)

    ra_schedule_df["RA release (EUR)"] = (
            ra_schedule_df["RA na začiatku roka (EUR)"].shift(1) - ra_schedule_df["RA na začiatku roka (EUR)"]
    )
    ra_schedule_df.loc[0, "RA release (EUR)"] = 0.0

    st.dataframe(
        ra_schedule_df.style.format({
            "BEL (EUR)": "{:,.2f}",
            "Sadzba RA (%)": "{:.4f}",
            "RA na začiatku roka (EUR)": "{:,.2f}",
            "RA release (EUR)": "{:,.2f}",
        }),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Grafy")

    import matplotlib.pyplot as plt
    import io

    def _fig_to_download(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()

    # --- Graf 1: BEL a RA po rokoch ---
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(ra_schedule_df["Rok"], ra_schedule_df["BEL (EUR)"] / 1e6, label="BEL základný", color="#1f77b4",
             linewidth=2)
    ax1.plot(ra_schedule_df["Rok"], ra_schedule_df["RA na začiatku roka (EUR)"] / 1e6, label="RA", color="#ff7f0e",
             linewidth=2, linestyle="--")
    ax1.set_xlabel("Rok")
    ax1.set_ylabel("EUR (mil.)")
    ax1.set_title("BEL a RA po rokoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1, use_container_width=False)
    st.download_button("Stiahnuť graf: BEL a RA po rokoch", _fig_to_download(fig1),
                       file_name="graf_bel_ra_po_rokoch.png", mime="image/png")
    plt.close(fig1)

    # --- Graf 2: NFR komponenty ---
    nfr_plot = scr_df[scr_df["Komponent"] != "CELKOM (agregované)"].copy()
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    bars = ax2.bar(nfr_plot["Komponent"], nfr_plot["NFR"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax2.set_ylabel("EUR")
    ax2.set_title("NFR komponenty")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(True, axis="y", alpha=0.3)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():,.0f}", ha="center",
                 va="bottom", fontsize=9)
    st.pyplot(fig2, use_container_width=False)
    st.download_button("Stiahnuť graf: NFR komponenty", _fig_to_download(fig2), file_name="graf_nfr_komponenty.png",
                       mime="image/png")
    plt.close(fig2)

    # --- Graf 3: RA release po rokoch ---
    release_plot = ra_schedule_df[ra_schedule_df["Rok"] > 0].copy()
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in release_plot["RA release (EUR)"]]
    ax3.bar(release_plot["Rok"], release_plot["RA release (EUR)"] / 1e3, color=colors)
    ax3.set_xlabel("Rok")
    ax3.set_ylabel("EUR (tis.)")
    ax3.set_title("RA release po rokoch")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig3, use_container_width=False)
    st.download_button("Stiahnuť graf: RA release", _fig_to_download(fig3), file_name="graf_ra_release.png",
                       mime="image/png")
    plt.close(fig3)

    # --- Graf 4: BEL šokovaný vs základný ---
    bel_plot = scr_df[scr_df["Komponent"] != "CELKOM (agregované)"].copy()
    x = range(len(bel_plot))
    width = 0.35
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    ax4.bar([i - width / 2 for i in x], [ra.bel_base / 1e6] * len(bel_plot), width, label="BEL základný",
            color="#1f77b4")
    ax4.bar([i + width / 2 for i in x],
            [ra.bel_stressed_by_component.get(k, ra.bel_base) / 1e6 for k in ra.scr_components.keys()], width,
            label="BEL šokovaný", color="#ff7f0e")
    ax4.set_xticks(list(x))
    ax4.set_xticklabels(bel_plot["Komponent"], rotation=15)
    ax4.set_ylabel("EUR (mil.)")
    ax4.set_title("BEL základný vs BEL šokovaný po komponentoch")
    ax4.legend()
    ax4.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig4, use_container_width=False)
    st.download_button("Stiahnuť graf: BEL základný vs šokovaný", _fig_to_download(fig4),
                       file_name="graf_bel_sokovany.png", mime="image/png")
    plt.close(fig4)

    st.subheader("Kontrola konzistencie")
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
        [{"Kontrola": c.name, "STAV": "OK" if c.passed else "CHYBA", "Detaily": c.details} for c in checks]
    )
    st.dataframe(chk_df, width="stretch", hide_index=True)

    st.subheader("Export")

    ra_export_df = pd.DataFrame([{
        "insurance_type": ra.insurance_type,
        "percentile": ra.percentile,
        "BEL_base": ra.bel_base,
        "RA_total": ra.ra_total,
        "RA_rate": ra.ra_rate,
        "Fulfillment_Cash_Flows": ra.bel_base + ra.ra_total,
    }])

    scr_export_df = pd.DataFrame([{
        "Komponent": nfr_names.get(k, k),
        "BEL šokovaný (EUR)": ra.bel_stressed_by_component.get(k, ra.bel_base),
        "NFR (EUR)": float(v),
    } for k, v in ra.scr_components.items()])

    ra_export_df.columns = [
        "Typ poistenia", "Percentil", "BEL základný", "Riziková prirážka RA",
        "Sadzba RA", "Peňažné toky na splnenie záväzkov (FCF)"
    ]

    ra_schedule_export = ra_schedule_df.copy()
    ra_schedule_export.columns = [
        "Rok", "BEL (EUR)", "Sadzba RA (%)", "RA na začiatku roka (EUR)", "RA release (EUR)"
    ]

    st.download_button(
        label="Stiahnuť ra_results.csv",
        data=ra_export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ra_results.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Stiahnuť nfr_components.csv",
        data=scr_export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="nfr_components.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Stiahnuť ra_po_rokoch.csv",
        data=ra_schedule_export.to_csv(index=False).encode("utf-8-sig"),
        file_name="ra_po_rokoch.csv",
        mime="text/csv",
    )