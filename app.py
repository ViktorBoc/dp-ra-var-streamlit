from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import io
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

DEFAULT_PREMIUM_RATE_IF_MISSING = 0.02  #2% of sum_insured

def _sk_fmt(x, decimals=2):
    """Slovenské formátovanie: medzera tisícky, desatinná čiarka."""
    if not isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, int) or decimals == 0:
        s = f"{abs(x):,.0f}".replace(",", "\u00a0")
        return f"-{s}" if x < 0 else s
    s = f"{abs(x):,.{decimals}f}".replace(",", "\u00a0").replace(".", ",")
    return f"-{s}" if x < 0 else s

def _table_to_png(df, title=""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 0.5 + len(df) * 0.4))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e6e6e6")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#cccccc")
    fig.tight_layout()
    return fig

def _fig_to_download(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

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

    fwd = rf["forward_rate"].to_numpy(dtype=float)
    if len(fwd) < 50:
        fwd = np.pad(fwd, (0, 50 - len(fwd)), mode="edge")
    fwd = fwd[:50]

    # Prémia za nelikviditu (Illiquidity Premium), bottom-up, +0,5 % k bezrizikovej spotovej sadzbe
    ILLIQUIDITY_PREMIUM = 0.005
    spot_illiquid = rf["spot_rate"].to_numpy(dtype=float) + ILLIQUIDITY_PREMIUM
    years = np.arange(1, len(spot_illiquid) + 1, dtype=float)
    df_illiquid = 1.0 / (1.0 + spot_illiquid) ** years
    if len(df_illiquid) < 50:
        df_illiquid = np.pad(df_illiquid, (0, 50 - len(df_illiquid)), mode="edge")
    df_illiquid = df_illiquid[:50]

    fwd_illiquid = np.empty(len(spot_illiquid), dtype=float)
    fwd_illiquid[0] = spot_illiquid[0]
    for i in range(1, len(spot_illiquid)):
        fwd_illiquid[i] = (
                                  (1.0 + spot_illiquid[i]) ** (i + 1) / (1.0 + spot_illiquid[i - 1]) ** i
                          ) - 1.0
    if len(fwd_illiquid) < 50:
        fwd_illiquid = np.pad(fwd_illiquid, (0, 50 - len(fwd_illiquid)), mode="edge")
    fwd_illiquid = fwd_illiquid[:50]

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
        forward_rates=fwd,
        discount_factors_illiquid=df_illiquid,
        forward_rates_illiquid=fwd_illiquid,
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
    - **FCF** (Fulfillment Cash Flows / Peňažné toky na splnenie záväzkov) – súčet BEL a rizikovej úpravy RA; celkový záväzok poisťovne vykazovaný podľa IFRS 17
    - **Coverage units** (Jednotky krytia) – miera poistnej služby poskytovanej v danom roku; v tejto aplikácii sa počítajú ako poistné sumy vážené podielom preživajúcich zmlúv
    - **PV** (Present Value / Súčasná hodnota) – hodnota budúcich peňažných tokov alebo veličín diskontovaná na dnešok pomocou bezrizikovej úrokovej miery
    - **PV CU** (Present Value of Coverage Units) – súčasná hodnota budúcich coverage units diskontovaná forwardovými sadzbami bezrizikovej krivky
    - **BoP** (Beginning of Period / Začiatok obdobia) – hodnota na začiatku roka
    - **EoP** (End of Period / Koniec obdobia) – hodnota na konci roka
    - **qx** – pravdepodobnosť úmrtia osoby vo veku x v priebehu jedného roka
    """)

port, assumptions, shock_engine, corr, product_map, var_levels = _load_and_prepare()
EXCLUDED_TYPES = {"UL_endowment"}
insurance_types = sorted([t for t in port["insurance_type"].unique().tolist() if t not in EXCLUDED_TYPES])

def _invalidate():
    st.session_state["do_compute"] = False

with st.sidebar:
    st.header("Vstupy")

    sk_names = {
        "term_insurance": "Poistenie pre prípad smrti (Term Insurance)",
        "whole_of_life": "Trvalé životné poistenie (Whole of Life)",
        "endowment": "Zmiešané poistenie (Endowment)",
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
        format_func=lambda x: f"{x * 100:.1f}%".replace(".", ",")
    )
    compute_btn = st.button("Vypočítať", type="primary")

if "do_compute" not in st.session_state:
    st.session_state["do_compute"] = False
if compute_btn:
    st.session_state["do_compute"] = True

# --- Súhrn portfólia (zobrazuje sa vždy po výbere typu) ---
port_preview = port[port["insurance_type"] == sel_type].copy()
port_preview = port_preview[port_preview["agreement_state"].isin(["new", "paid_up"])]

if len(port_preview) > 0:
    st.subheader("Súhrn portfólia")

    from datetime import date
    valuation_date = date(2026, 1, 1)
    port_preview["vek"] = port_preview["date_of_birth"].apply(
        lambda x: int((valuation_date - pd.to_datetime(x).date()).days / 365.25)
    )

    count_new = int((port_preview["agreement_state"] == "new").sum())
    count_paid = int((port_preview["agreement_state"] == "paid_up").sum())
    avg_age = float(port_preview["vek"].mean())
    avg_sum = float(port_preview["sum_insured"].mean())
    avg_prem = float(port_preview["premium"].mean()) if "premium" in port_preview.columns else float(port_preview["annual_premium"].mean())

    summary_df = pd.DataFrame([
        {"Ukazovateľ": "Počet zmlúv celkom", "Hodnota": f"{count_new + count_paid}"},
        {"Ukazovateľ": "Z toho new", "Hodnota": f"{count_new}"},
        {"Ukazovateľ": "Z toho paid_up", "Hodnota": f"{count_paid}"},
        {"Ukazovateľ": "Priemerný vek poisteného", "Hodnota": f"{_sk_fmt(avg_age, 1)} rokov"},
        {"Ukazovateľ": "Priemerná poistná suma" if sel_type != "annuity" else "Priemerná ročná renta",
        "Hodnota": f"{_sk_fmt(avg_sum, 0)} EUR"},
        {"Ukazovateľ": "Priemerné ročné poistné", "Hodnota": f"{_sk_fmt(avg_prem, 0)} EUR"},
    ])
    st.dataframe(summary_df, hide_index=True, width="stretch")
    _fig_port = _table_to_png(summary_df, "Súhrn portfólia")
    st.download_button("Stiahnuť tabuľku: Súhrn portfólia", _fig_to_download(_fig_port),
                       file_name="tab_suhrn_portfolia.png", mime="image/png", key="dl_tab_portfolio")
    import matplotlib.pyplot as plt
    plt.close(_fig_port)

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
    c1.metric("Riziková úprava z nefinančného rizika - RA (EUR)", _sk_fmt(ra.ra_total, 2))
    c2.metric("BEL základný (EUR)", _sk_fmt(ra.bel_base, 2))
    c3.metric("Peňažné toky na splnenie záväzkov - FCF (EUR)", _sk_fmt(ra.bel_base + ra.ra_total, 2))
    st.markdown("**NFR komponenty (portfólio)**")
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
            "BEL šokovaný (EUR)": lambda x: _sk_fmt(x, 2) if isinstance(x, (int, float)) else x,
            "NFR": lambda x: _sk_fmt(x, 2),
        }),
        width="stretch",
        hide_index=True,
    )
    _scr_display = scr_df.copy()
    _scr_display["BEL šokovaný (EUR)"] = _scr_display["BEL šokovaný (EUR)"].apply(
        lambda x: _sk_fmt(x, 2) if isinstance(x, (int, float)) else x)
    _scr_display["NFR"] = _scr_display["NFR"].apply(lambda x: _sk_fmt(x, 2))
    _fig_nfr = _table_to_png(_scr_display, "NFR komponenty (portfólio)")
    st.download_button("Stiahnuť tabuľku: NFR komponenty", _fig_to_download(_fig_nfr),
                       file_name="tab_nfr_komponenty.png", mime="image/png", key="dl_tab_nfr")
    plt.close(_fig_nfr)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**RA po rokoch (rozpustenie)**")

    from src.cashflows import portfolio_coverage_units

    max_horizon = max((p.horizon for p in policies), default=0)
    max_years = min(max_horizon, 50)

    # Coverage units pre každý rok
    cu = portfolio_coverage_units(policies, assumptions, max_years)

    # Forward rates: nelikvidné produkty (endowment, annuity) používajú krivku s IP
    if sel_type in ("endowment", "annuity"):
        fwd = assumptions.forward_rates_illiquid[:max_years]
    else:
        fwd = assumptions.forward_rates[:max_years]

    # PV budúcich coverage units pre každý rok t
    pv_cu = np.zeros(max_years, dtype=float)
    for t in range(max_years):
        pv = 0.0
        disc = 1.0
        for s in range(t, max_years):
            if s > t:
                disc /= (1.0 + fwd[s])
            pv += cu[s] * disc
        pv_cu[t] = pv

    # Amortizačný faktor = CU(t) / PV_CU(t)
    amort = np.zeros(max_years, dtype=float)
    for t in range(max_years):
        amort[t] = (cu[t] / pv_cu[t]) if pv_cu[t] > 1e-12 else 1.0

    # Rozpúšťanie RA cez coverage units
    ra_bop = np.zeros(max_years, dtype=float)
    ra_release = np.zeros(max_years, dtype=float)
    ra_eop = np.zeros(max_years, dtype=float)

    ra_bop[0] = ra.ra_total
    for t in range(max_years):
        ra_release[t] = amort[t] * ra_bop[t]
        ra_eop[t] = ra_bop[t] - ra_release[t]
        if t + 1 < max_years:
            ra_bop[t + 1] = ra_eop[t]

    rows = []
    for t in range(max_years):
        rows.append({
            "Rok": t + 1,
            "Poistná suma – Coverage units (EUR)": cu[t],
            "PV poistnej sumy – PV CU (EUR)": pv_cu[t],
            "Amortizačný faktor (%)": amort[t] * 100,
            "RA BoP (EUR)": ra_bop[t],
            "Rozpustenie RA (EUR)": ra_release[t],
            "RA EoP (EUR)": ra_eop[t],
        })

    ra_schedule_df = pd.DataFrame(rows)

    st.dataframe(
        ra_schedule_df.style.format({
            "Poistná suma – Coverage units (EUR)": lambda x: _sk_fmt(x, 0),
            "PV poistnej sumy – PV CU (EUR)": lambda x: _sk_fmt(x, 0),
            "Amortizačný faktor (%)": lambda x: _sk_fmt(x, 2),
            "RA BoP (EUR)": lambda x: _sk_fmt(x, 2),
            "Rozpustenie RA (EUR)": lambda x: _sk_fmt(x, 2),
            "RA EoP (EUR)": lambda x: _sk_fmt(x, 2),
        }),
        width="stretch",
        hide_index=True,
    )
    _ra_sch_display = ra_schedule_df.copy()
    for col in ["Poistná suma – Coverage units (EUR)", "PV poistnej sumy – PV CU (EUR)"]:
        _ra_sch_display[col] = _ra_sch_display[col].apply(lambda x: _sk_fmt(x, 0))
    _ra_sch_display["Amortizačný faktor (%)"] = _ra_sch_display["Amortizačný faktor (%)"].apply(lambda x: _sk_fmt(x, 2))
    for col in ["RA BoP (EUR)", "Rozpustenie RA (EUR)", "RA EoP (EUR)"]:
        _ra_sch_display[col] = _ra_sch_display[col].apply(lambda x: _sk_fmt(x, 2))
    _fig_rasch = _table_to_png(_ra_sch_display, "RA po rokoch (rozpustenie)")
    st.download_button("Stiahnuť tabuľku: RA po rokoch", _fig_to_download(_fig_rasch),
                       file_name="tab_ra_po_rokoch.png", mime="image/png", key="dl_tab_ra_roky")
    plt.close(_fig_rasch)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Grafy a tabuľky")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Súhrnné výsledky**")
    from src.cashflows import portfolio_bel as _pbel, Scenario as _Scenario
    _base_result = _pbel(policies, assumptions, _Scenario())
    from datetime import date as _date

    _vd = _date(2026, 1, 1)
    _port_c = port_sel[port_sel["agreement_state"].isin(["new", "paid_up"])].copy()
    _port_c["_vek"] = _port_c["date_of_birth"].apply(
        lambda x: int(((_vd - pd.to_datetime(x).date()).days) / 365.25))
    _cnt_new_c = int((_port_c["agreement_state"] == "new").sum())
    _cnt_paid_c = int((_port_c["agreement_state"] == "paid_up").sum())
    _avg_age_c = float(_port_c["_vek"].mean()) if len(_port_c) > 0 else 0.0
    _avg_sum_c = float(_port_c["sum_insured"].mean()) if len(_port_c) > 0 else 0.0
    _avg_prem_c = float(_port_c["premium"].mean()) if len(_port_c) > 0 else 0.0
    _sum_label_c = "Priemerná ročná renta (EUR)" if sel_type == "annuity" else "Priemerná poistná suma (EUR)"

    summary_results = pd.DataFrame([
        {"Ukazovateľ": "Typ poistenia", "Hodnota": sk_names.get(ra.insurance_type, ra.insurance_type)},
        {"Ukazovateľ": "Percentil", "Hodnota": f"{_sk_fmt(ra.percentile * 100, 1)}%"},
        {"Ukazovateľ": "Počet zmlúv celkom", "Hodnota": f"{len(policies)}"},
        {"Ukazovateľ": "Z toho new", "Hodnota": f"{_cnt_new_c}"},
        {"Ukazovateľ": "Z toho paid_up", "Hodnota": f"{_cnt_paid_c}"},
        {"Ukazovateľ": "Priemerný vek poisteného", "Hodnota": f"{_sk_fmt(_avg_age_c, 1)} rokov"},
        {"Ukazovateľ": _sum_label_c, "Hodnota": _sk_fmt(_avg_sum_c, 0)},
        {"Ukazovateľ": "Priemerné ročné poistné (EUR)", "Hodnota": _sk_fmt(_avg_prem_c, 0)},
        {"Ukazovateľ": "Súčasná hodnota peňažných príjmov (EUR)", "Hodnota": _sk_fmt(_base_result["pv_inflows"], 2)},
        {"Ukazovateľ": "Súčasná hodnota peňažných výdavkov (EUR)", "Hodnota": _sk_fmt(_base_result["pv_outflows"], 2)},
        {"Ukazovateľ": "BEL základný (EUR)", "Hodnota": _sk_fmt(ra.bel_base, 2)},
        {"Ukazovateľ": "Riziková úprava – RA (EUR)", "Hodnota": _sk_fmt(ra.ra_total, 2)},
        {"Ukazovateľ": "Peňažné toky na splnenie záväzkov – FCF (EUR)",
         "Hodnota": _sk_fmt(ra.bel_base + ra.ra_total, 2)},
    ])
    st.dataframe(summary_results, hide_index=True, width="stretch")

    _fig_sr = _table_to_png(summary_results, "Súhrnné výsledky")
    st.download_button("Stiahnuť tabuľku: Súhrnné výsledky", _fig_to_download(_fig_sr),
                       file_name="tab_suhrnne_vysledky.png", mime="image/png", key="dl_tab_suhrn")
    with st.expander("ℹ️ Popis tabuľky"):
        st.markdown(
            f"**Tabuľka:** Súhrnné výsledky  \n**Produkt:** {sk_names.get(sel_type, sel_type)}  \n**Percentil:** {_sk_fmt(sel_p * 100, 1)}%  \n**Popis:** Kľúčové ukazovatele portfólia vrátane štruktúry zmlúv, priemerných charakteristík poistených, najlepšieho odhadu záväzkov BEL, rizikovej úpravy RA a celkových peňažných tokov na splnenie záväzkov FCF podľa IFRS 17.")
    plt.close(_fig_sr)

    # Vypočítaj RA pre všetky percentily
    ra_by_p = []
    for p_val in [float(x) for x in var_levels]:
        r_p = compute_ra(
            insurance_type=sel_type,
            percentile=p_val,
            policies=policies,
            assumptions=assumptions,
            corr=corr,
            product_components=components,
            shock_engine=shock_engine,
            index_base=assumptions.index_base,
            index_stressed=assumptions.index_stressed,
        )
        ra_by_p.append((p_val, float(r_p.ra_total)))

    # --- Graf: Poistná suma (Coverage units) po rokoch ---
    st.markdown("<br>", unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig1a, ax1a = plt.subplots(figsize=(5, 3))
        ax1a.plot(ra_schedule_df["Rok"], ra_schedule_df["Poistná suma – Coverage units (EUR)"] / 1e6,
                    color="#1f77b4", linewidth=2)
        ax1a.set_xlabel("Rok")
        ax1a.set_ylabel("EUR (mil.)")
        ax1a.set_title("Poistná suma (Coverage units) po rokoch")
        ax1a.grid(True, alpha=0.3)
        fig1a.tight_layout()
        st.pyplot(fig1a, use_container_width=True)
        st.download_button("Stiahnuť graf: Coverage units", _fig_to_download(fig1a),
                           file_name="graf_coverage_units.png", mime="image/png", key="dl_graf_coverage_units")
        with st.expander("ℹ️ Popis grafu"):
            st.markdown(
                f"**Graf:** Poistná suma – Coverage units po rokoch  \n**Produkt:** {sk_names.get(sel_type, sel_type)}  \n**Percentil:** {_sk_fmt(sel_p * 100, 1)}%  \n**Popis:** Vývoj poistných súm vážených podielom preživajúcich zmlúv počas projekčného horizontu portfólia.")
        plt.close(fig1a)

    # --- Graf: RA po rokoch ---
    with col_g2:
        fig1b, ax1b = plt.subplots(figsize=(5, 3))
        ax1b.plot(ra_schedule_df["Rok"], ra_schedule_df["RA BoP (EUR)"] / 1e3,
                    color="#1f77b4", linewidth=2)
        ax1b.set_xlabel("Rok")
        ax1b.set_ylabel("EUR (tis.)")
        ax1b.set_title("RA na začiatku roka (BoP)")
        ax1b.grid(True, alpha=0.3)
        fig1b.tight_layout()
        st.pyplot(fig1b, use_container_width=True)
        st.download_button("Stiahnuť graf: RA BoP po rokoch", _fig_to_download(fig1b),
                           file_name="graf_ra_po_rokoch.png", mime="image/png", key="dl_graf_ra_bop_roky")
        with st.expander("ℹ️ Popis grafu"):
            st.markdown(
                f"**Graf:** RA na začiatku roka (BoP) po rokoch  \n**Produkt:** {sk_names.get(sel_type, sel_type)}  \n**Percentil:** {_sk_fmt(sel_p * 100, 1)}%  \n**Popis:** Zostatok rizikovej úpravy RA na začiatku každého roka po postupnom rozpúšťaní cez coverage units.")
        plt.close(fig1b)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # --- Graf: NFR komponenty ---
    nfr_plot = scr_df[scr_df["Komponent"] != "CELKOM (agregované)"].copy()
    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    bars = ax2.bar(nfr_plot["Komponent"], nfr_plot["NFR"] / 1e3,
                   color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax2.set_ylabel("EUR (tis.)")
    ax2.set_title("NFR komponenty")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(True, axis="y", alpha=0.3)
    for bar in bars:
        val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, val,
                 _sk_fmt(val, 1), ha="center", va="bottom", fontsize=9)
    st.pyplot(fig2, use_container_width=False)
    st.download_button("Stiahnuť graf: NFR komponenty", _fig_to_download(fig2),
                       file_name="graf_nfr_komponenty.png", mime="image/png")
    with st.expander("ℹ️ Popis grafu"):
        st.markdown(
            f"**Graf:** NFR komponenty  \n**Produkt:** {sk_names.get(sel_type, sel_type)}  \n**Percentil:** {_sk_fmt(sel_p * 100, 1)}%  \n**Popis:** Výška nefinančného rizikového kapitálu (NFR) pre jednotlivé rizikové komponenty pred agregáciou cez korelačnú maticu.")
    plt.close(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    # --- Graf: RA release a Amortizačný faktor vedľa seba ---
    col_g3, col_g4 = st.columns(2)

    with col_g3:
        release_plot = ra_schedule_df[ra_schedule_df["Rok"] > 0].copy()
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in release_plot["Rozpustenie RA (EUR)"]]
        ax3.bar(release_plot["Rok"], release_plot["Rozpustenie RA (EUR)"] / 1e3, color=colors)
        ax3.set_xlabel("Rok")
        ax3.set_ylabel("EUR (tis.)")
        ax3.set_title("Rozpustenie RA po rokoch")
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.grid(True, axis="y", alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        st.download_button("Stiahnuť graf: Rozpustenie RA", _fig_to_download(fig3),
                           file_name="graf_ra_release.png", mime="image/png", key="dl_graf_ra_release")
        with st.expander("ℹ️ Popis grafu"):
            st.markdown(
                f"**Graf:** Rozpustenie RA po rokoch  \n**Produkt:** {sk_names.get(sel_type, sel_type)}  \n**Percentil:** {_sk_fmt(sel_p * 100, 1)}%  \n**Popis:** Objem rizikovej úpravy RA rozpustenej (vykázanej do výnosov) v každom roku podľa amortizačného faktora.")
        plt.close(fig3)

    with col_g4:
        fig_af, ax_af = plt.subplots(figsize=(5, 3))
        ax_af.plot(ra_schedule_df["Rok"], ra_schedule_df["Amortizačný faktor (%)"],
                   color="#1f77b4", linewidth=2, marker=".")
        ax_af.set_xlabel("Rok")
        ax_af.set_ylabel("%")
        ax_af.set_title("Amortizačný faktor po rokoch")
        ax_af.grid(True, alpha=0.3)
        fig_af.tight_layout()
        st.pyplot(fig_af, use_container_width=True)
        st.download_button("Stiahnuť graf: Amortizačný faktor", _fig_to_download(fig_af),
                           file_name="graf_amortizacny_faktor.png", mime="image/png", key="dl_graf_amort")
        with st.expander("ℹ️ Popis grafu"):
            st.markdown(
                f"**Graf:** Amortizačný faktor po rokoch  \n**Produkt:** {sk_names.get(sel_type, sel_type)}  \n**Percentil:** {_sk_fmt(sel_p * 100, 1)}%  \n**Popis:** Pomer coverage units v danom roku k súčasnej hodnote všetkých zostávajúcich coverage units; určuje, aký podiel RA sa v danom roku rozpustí.")
        plt.close(fig_af)

    st.markdown("<br>", unsafe_allow_html=True)
    # --- Graf: RA podľa percentilov ---
    p_labels = ["99,5%" if p == 0.995 else f"{p * 100:.0f}%" for p, _ in ra_by_p]
    ra_values = [v / 1e3 for _, v in ra_by_p]

    fig5, ax5 = plt.subplots(figsize=(4, 3))
    ax5.plot(p_labels, ra_values, color="#1f77b4", linewidth=2, marker="o")
    ax5.set_xlabel("Percentil")
    ax5.set_ylabel("EUR (tis.)")
    ax5.set_title("RA podľa percentilov (EUR)")
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    st.pyplot(fig5, use_container_width=False)
    st.download_button("Stiahnuť graf: RA podľa percentilov", _fig_to_download(fig5),
                       file_name="graf_ra_percentily.png", mime="image/png")
    with st.expander("ℹ️ Popis grafu"):
        st.markdown(
            f"**Graf:** RA podľa percentilov  \n**Produkt:** {sk_names.get(sel_type, sel_type)}  \n**Percentil:** všetky ({', '.join(['99,5%' if p == 0.995 else f'{p * 100:.0f}%' for p, _ in ra_by_p])})  \n**Popis:** Závislosť celkovej rizikovej úpravy RA od zvoleného percentilu spoľahlivosti VaR.")
    plt.close(fig5)

    st.markdown("**RA podľa percentilov**")
    ra_perc_table = pd.DataFrame([{
        "Percentil": f"{_sk_fmt(p * 100, 1)}%",
        "RA (EUR)": v,
        "BEL základný (EUR)": ra.bel_base,
        "FCF (EUR)": ra.bel_base + v,
    } for p, v in ra_by_p])
    ra_perc_table = pd.concat([ra_perc_table, pd.DataFrame([{
        "Percentil": "",
        "RA (EUR)": "",
        "BEL základný (EUR)": "",
        "FCF (EUR)": "",
    }, {
        "Percentil": "Typ poistenia",
        "RA (EUR)": sk_names.get(sel_type, sel_type),
        "BEL základný (EUR)": "",
        "FCF (EUR)": "",
    }, {
        "Percentil": "Počet zmlúv celkom",
        "RA (EUR)": str(len(policies)),
        "BEL základný (EUR)": "",
        "FCF (EUR)": "",
    }, {
        "Percentil": "Z toho new",
        "RA (EUR)": str(_cnt_new_c),
        "BEL základný (EUR)": "",
        "FCF (EUR)": "",
    }, {
        "Percentil": "Z toho paid_up",
        "RA (EUR)": str(_cnt_paid_c),
        "BEL základný (EUR)": "",
        "FCF (EUR)": "",
    }, {
        "Percentil": "Priemerný vek poisteného",
        "RA (EUR)": f"{_sk_fmt(_avg_age_c, 1)} rokov",
        "BEL základný (EUR)": "",
        "FCF (EUR)": "",
    }, {
        "Percentil": _sum_label_c,
        "RA (EUR)": _sk_fmt(_avg_sum_c, 0),
        "BEL základný (EUR)": "",
        "FCF (EUR)": "",
    }, {
        "Percentil": "Priemerné ročné poistné (EUR)",
        "RA (EUR)": _sk_fmt(_avg_prem_c, 0),
        "BEL základný (EUR)": "",
        "FCF (EUR)": "",
    }])], ignore_index=True)
    st.dataframe(
        ra_perc_table.style.format({
            "RA (EUR)": lambda x: _sk_fmt(x, 2) if isinstance(x, (int, float)) else x,
            "BEL základný (EUR)": lambda x: _sk_fmt(x, 2) if isinstance(x, (int, float)) else x,
            "FCF (EUR)": lambda x: _sk_fmt(x, 2) if isinstance(x, (int, float)) else x,
        }),
        hide_index=True, width="stretch",
    )
    _perc_display = ra_perc_table.copy()
    for col in ["RA (EUR)", "BEL základný (EUR)", "FCF (EUR)"]:
        _perc_display[col] = _perc_display[col].apply(
            lambda x: _sk_fmt(x, 2) if isinstance(x, (int, float)) else x)
    _fig_perc = _table_to_png(_perc_display, "RA podľa percentilov")
    st.download_button("Stiahnuť tabuľku: RA podľa percentilov", _fig_to_download(_fig_perc),
                       file_name="tab_ra_percentily.png", mime="image/png", key="dl_tab_ra_perc")
    with st.expander("ℹ️ Popis tabuľky"):
        st.markdown(
            f"**Tabuľka:** RA podľa percentilov  \n**Produkt:** {sk_names.get(sel_type, sel_type)}  \n**Percentil:** všetky  \n**Popis:** Výška rizikovej úpravy RA, najlepší odhad záväzkov BEL a celkové peňažné toky na splnenie záväzkov FCF pre každý percentil spoľahlivosti VaR. Zahŕňa aj kľúčové charakteristiky portfólia.")
    plt.close(_fig_perc)

    st.markdown("<br>", unsafe_allow_html=True)
    # --- Graf: RA podľa produktov (EUR) ---
    st.markdown("**Porovnanie produktov**")
    all_types = [t for t in port["insurance_type"].unique().tolist() if t not in EXCLUDED_TYPES]
    ra_by_product = []
    for prod in all_types:
        port_prod = port[port["insurance_type"] == prod].copy()
        pol_prod = _build_policies(port_prod)
        if not pol_prod:
            continue
        comp_prod = product_map.get(prod, {}).get("components", ["mortality", "longevity", "lapse", "expense"])
        ra_prod = compute_ra(
            insurance_type=prod,
            percentile=float(sel_p),
            policies=pol_prod,
            assumptions=assumptions,
            corr=corr,
            product_components=comp_prod,
            shock_engine=shock_engine,
            index_base=assumptions.index_base,
            index_stressed=assumptions.index_stressed,
        )
        ra_by_product.append((prod, float(ra_prod.ra_total) / 1e3))

    fig6, ax6 = plt.subplots(figsize=(5.5, 4.5))
    prod_labels = [sk_names.get(p, p) for p, _ in ra_by_product]
    prod_eur = [e for _, e in ra_by_product]

    ax6.bar(prod_labels, prod_eur, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax6.set_ylabel("EUR (tis.)")
    ax6.set_title("RA podľa produktov")
    ax6.tick_params(axis="x", rotation=15)
    ax6.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(ax6.patches, prod_eur):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 _sk_fmt(val, 1), ha="center", va="bottom", fontsize=9)
    fig6.tight_layout()
    st.pyplot(fig6, use_container_width=False)
    st.download_button("Stiahnuť graf: Porovnanie produktov", _fig_to_download(fig6),
                       file_name="graf_porovnanie_produktov.png", mime="image/png")
    with st.expander("ℹ️ Popis grafu"):
        st.markdown(
            f"**Graf:** RA podľa produktov  \n**Produkt:** všetky  \n**Percentil:** {_sk_fmt(sel_p * 100, 1)}%  \n**Popis:** Porovnanie výšky rizikovej úpravy RA naprieč všetkými typmi poistenia pri vybranom percentile.")
    plt.close(fig6)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**RA podľa produktov**")
    ra_prod_table_rows = []
    for prod, ra_eur in ra_by_product:
        port_p = port[port["insurance_type"] == prod]
        port_p = port_p[port_p["agreement_state"].isin(["new", "paid_up"])].copy()
        comp_prod = product_map.get(prod, {}).get("components", ["mortality", "longevity", "lapse", "expense"])
        ra_prod_full = compute_ra(
            insurance_type=prod,
            percentile=float(sel_p),
            policies=_build_policies(port_p),
            assumptions=assumptions,
            corr=corr,
            product_components=comp_prod,
            shock_engine=shock_engine,
            index_base=assumptions.index_base,
            index_stressed=assumptions.index_stressed,
        )
        _pp_vd = _date(2026, 1, 1)
        port_p["_vek_p"] = port_p["date_of_birth"].apply(
            lambda x: int(((_pp_vd - pd.to_datetime(x).date()).days) / 365.25))
        _p_cnt_new = int((port_p["agreement_state"] == "new").sum())
        _p_cnt_paid = int((port_p["agreement_state"] == "paid_up").sum())
        _p_avg_age = float(port_p["_vek_p"].mean()) if len(port_p) > 0 else 0.0
        _p_avg_sum = float(port_p["sum_insured"].mean()) if len(port_p) > 0 else 0.0
        _p_avg_prem = float(port_p["premium"].mean()) if len(port_p) > 0 else 0.0
        ra_prod_table_rows.append({
            "Produkt": sk_names.get(prod, prod),
            "Počet zmlúv": int(len(port_p)),
            "Z toho new": _p_cnt_new,
            "Z toho paid_up": _p_cnt_paid,
            "Priemerný vek": f"{_sk_fmt(_p_avg_age, 1)} r.",
            "Priem. suma / renta (EUR)": _sk_fmt(_p_avg_sum, 0),
            "Priem. ročné poistné (EUR)": _sk_fmt(_p_avg_prem, 0),
            "BEL základný (EUR)": ra_prod_full.bel_base,
            "RA (EUR)": ra_eur * 1e3,
            "FCF (EUR)": ra_prod_full.bel_base + ra_eur * 1e3,
        })
    ra_prod_table = pd.DataFrame(ra_prod_table_rows)
    st.dataframe(
        ra_prod_table.style.format({
            "BEL základný (EUR)": lambda x: _sk_fmt(x, 2),
            "RA (EUR)": lambda x: _sk_fmt(x, 2),
            "FCF (EUR)": lambda x: _sk_fmt(x, 2),
        }),
        hide_index=True, width="stretch",
    )
    _prod_display = ra_prod_table.copy()
    for col in ["BEL základný (EUR)", "RA (EUR)", "FCF (EUR)"]:
        _prod_display[col] = _prod_display[col].apply(lambda x: _sk_fmt(x, 2))
    _fig_prod = _table_to_png(_prod_display, "RA podľa produktov")
    st.download_button("Stiahnuť tabuľku: RA podľa produktov", _fig_to_download(_fig_prod),
                       file_name="tab_ra_produkty.png", mime="image/png", key="dl_tab_ra_prod")
    with st.expander("ℹ️ Popis tabuľky"):
        st.markdown(
            f"**Tabuľka:** RA podľa produktov  \n**Produkt:** všetky  \n**Percentil:** {_sk_fmt(sel_p * 100, 1)}%  \n**Popis:** Porovnanie rizikovej úpravy RA, BEL a FCF naprieč všetkými typmi poistenia pri vybranom percentile spoľahlivosti VaR. Zahŕňa štruktúru portfólia (new / paid_up) a priemerné charakteristiky zmlúv.")
    plt.close(_fig_prod)

    # --- Graf: Heatmapa RA (EUR) – produkty × percentily ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Heatmapa RA (produkty × percentily)**")
    st.info("Výpočet heatmapy prebieha pre všetky produkty a percentily – môže trvať chvíľu.")
    heatmap_data = {}
    for prod in all_types:
        port_prod = port[port["insurance_type"] == prod].copy()
        pol_prod = _build_policies(port_prod)
        if not pol_prod:
            continue
        comp_prod = product_map.get(prod, {}).get("components", ["mortality", "longevity", "lapse", "expense"])
        vals = []
        for p_val in [float(x) for x in var_levels]:
            ra_h = compute_ra(
                insurance_type=prod,
                percentile=p_val,
                policies=pol_prod,
                assumptions=assumptions,
                corr=corr,
                product_components=comp_prod,
                shock_engine=shock_engine,
                index_base=assumptions.index_base,
                index_stressed=assumptions.index_stressed,
            )
            vals.append(float(ra_h.ra_total) / 1e3)
        heatmap_data[sk_names.get(prod, prod)] = vals

    p_labels_hm = ["99,5%" if p == 0.995 else f"{p * 100:.0f}%" for p in [float(x) for x in var_levels]]
    heatmap_df = pd.DataFrame(heatmap_data, index=p_labels_hm).T

    fig7, ax7 = plt.subplots(figsize=(10, 3))
    im = ax7.imshow(heatmap_df.values, aspect="auto", cmap="YlOrRd")
    ax7.set_xticks(range(len(p_labels_hm)))
    ax7.set_xticklabels(p_labels_hm)
    ax7.set_yticks(range(len(heatmap_df.index)))
    ax7.set_yticklabels(heatmap_df.index)
    ax7.set_title("RA (tis. EUR) – produkty × percentily")
    plt.colorbar(im, ax=ax7, label="RA (tis. EUR)")
    for i in range(len(heatmap_df.index)):
        for j in range(len(p_labels_hm)):
            ax7.text(j, i, _sk_fmt(heatmap_df.values[i, j], 1),
                     ha="center", va="center", fontsize=8, color="black")
    fig7.tight_layout()
    st.pyplot(fig7, use_container_width=False)
    st.download_button("Stiahnuť graf: Heatmapa RA", _fig_to_download(fig7),
                       file_name="graf_heatmapa_ra.png", mime="image/png")
    with st.expander("ℹ️ Popis grafu"):
        st.markdown(
            "**Graf:** Heatmapa RA (tis. EUR) – produkty × percentily  \n**Produkt:** všetky  \n**Percentil:** všetky  \n**Popis:** Mapa rizikovej úpravy RA pre všetky kombinácie typov poistenia a percentilov spoľahlivosti.")
    plt.close(fig7)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Kontrola konzistencie**")
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

    st.markdown("**Export**")

    ra_export_df = pd.DataFrame([{
        "insurance_type": ra.insurance_type,
        "percentile": ra.percentile,
        "BEL_base": ra.bel_base,
        "RA_total": ra.ra_total,
        "Fulfillment_Cash_Flows": ra.bel_base + ra.ra_total,
    }])

    scr_export_df = pd.DataFrame([{
        "Komponent": nfr_names.get(k, k),
        "BEL šokovaný (EUR)": ra.bel_stressed_by_component.get(k, ra.bel_base),
        "NFR (EUR)": float(v),
    } for k, v in ra.scr_components.items()])

    ra_export_df.columns = [
        "Typ poistenia", "Percentil", "BEL základný", "Riziková úprava z nefinančného rizika - RA (EUR)",
        "Peňažné toky na splnenie záväzkov - FCF (EUR)"
    ]

    ra_schedule_export = ra_schedule_df.copy()
    ra_schedule_export.columns = [
        "Rok", "Poistná suma – Coverage units (EUR)", "PV poistnej sumy – PV CU (EUR)", "Amortizačný faktor (%)",
        "RA BoP (EUR)", "Rozpustenie RA (EUR)", "RA EoP (EUR)"
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