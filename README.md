# Risk Adjustment (VaR) pre životné poistenie (Streamlit)

Tento projekt je webová aplikácia v **Streamlit**, ktorá pre vybrané portfólio zmlúv vypočíta **Risk Adjustment (RA)** metódou **VaR** podľa štandardu **IFRS 17**. Rizikové komponenty sa agregujú cez **variance–covariance** prístup (korelačná matica). Všetko funguje iba z lokálnych súborov (žiadne externé DB ani internetové API).

---

## Čo appka počíta (výstupy)

Používateľ vyberie:
- `insurance_type` (typ poistenia / produktu)
- percentil `p` z `{0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.995}` (načítané z `data/config/var_levels.json`)

Appka vypočíta pre portfólio všetkých zmlúv daného typu (len aktívne stavy):

**BEL základný (EUR)** = súčasná hodnota peňažných výdavkov − súčasná hodnota peňažných príjmov
- **Riziková úprava RA (EUR)** = agregované NFR komponenty cez korelačnú maticu
- **Peňažné toky na splnenie záväzkov – FCF (EUR)** = BEL_base + RA_total (IFRS 17 záväzok)
- **Súhrnná tabuľka**: typ poistenia, percentil, počet zmlúv celkom (z toho `new` / `paid_up`), priemerný vek poisteného, priemerná poistná suma alebo ročná renta, priemerné ročné poistné, súčasná hodnota peňažných príjmov a výdavkov, BEL, RA, FCF
- **Tabuľka RA podľa percentilov**: RA (EUR), BEL a FCF pre všetky percentily pri vybranom produkte; tabuľka obsahuje aj kľúčové charakteristiky portfólia (počet zmlúv, z toho `new` / `paid_up`, priemerný vek, priemerná poistná suma / renta, priemerné ročné poistné)
- **Tabuľka RA podľa produktov**: RA (EUR), BEL a FCF pre všetky produkty pri vybranom percentile; pre každý produkt zahŕňa počet zmlúv (z toho `new` / `paid_up`), priemerný vek poisteného, priemernú poistnú sumu alebo ročnú rentu a priemerné ročné poistné
- **NFR komponenty (EUR)**: `mortality`, `longevity`, `lapse`, `expense` vrátane BEL šokovaného pre každý komponent
- **RA po rokoch** = tabuľka rozpustenia RA počas trvania portfólia; stĺpce: `Rok`, `Poistná suma – Coverage units (EUR)`, `PV poistnej sumy – PV CU (EUR)`, `Amortizačný faktor (%)`, `RA BoP (EUR)`, `Rozpustenie RA (EUR)`, `RA EoP (EUR)` – podrobný popis výpočtu v sekcii [RA po rokoch](#ra-po-rokoch-rozpustenie-cez-coverage-units)
- **Grafy**: Coverage units po rokoch, RA BoP po rokoch, NFR komponenty, rozpustenie RA po rokoch, amortizačný faktor po rokoch, RA podľa percentilov (EUR), RA podľa produktov (EUR), heatmapa RA (EUR tis.) produkty × percentily; každý graf obsahuje rozbaľovací popis s informáciou o type, produkte a percentile
- **Kontrola konzistencie** (OK/CHYBA)

Export:
- `ra_results.csv` – súhrnné výsledky RA
- `nfr_components.csv` – NFR komponenty s BEL šokovaným
- `ra_po_rokoch.csv` – rozpustenie RA po rokoch
- Grafy stiahnuteľné ako PNG

---

## Kľúčové pravidlá

### Portfólio (filter)
- zahrnuté len `agreement_state ∈ {new, paid_up}`
- `declined`, `closed` ignorované
- `UL_endowment` je vylúčený z UI

### Valuation date a projekčný horizont
- valuation date je fixný: **2026-01-01**
- výpočet veku a trvania (konvencia 365.25 dní):
  - `current_age = floor((valuation_date - date_of_birth).days / 365.25)`
  - `duration = floor((valuation_date - issue_date).days / 365.25)`
  - `remaining_term = max(0, insurance_term - duration)`
  - ak `insurance_term == 9999` → whole_of_life, horizont sa zreže na max 50 rokov
- `max_age = 105`
- horizont na zmluvu: `T_i = min(remaining_term, 50, max_age - current_age)`

### Premium payment term
- každá zmluva má `premium_payment_term` – počet rokov platenia poistného
- v kóde sa ošetruje: `ppt = min(premium_payment_term, insurance_term)` – klient nemôže platiť dlhšie ako trvá zmluva
- `remaining_ppt = max(0, ppt - duration)` – zostatok doby platenia k dátumu ocenenia
- peňažné príjmy (poistné) sa zastavia po uplynutí `remaining_ppt` rokov projekcie

### Surrender Value (odkupná hodnota)
- pri každom storne (lapse) sa vypláca odkupná hodnota klientovi
- implementovaná ako **lineárna interpolácia klesajúca k nule**:
  - `sv(t) = surrender_value * max(0, horizon - t) / horizon`
  - v roku 0 = plná odkupná hodnota, v poslednom roku = 0
- hodnota `surrender_value` sa načítava priamo zo stĺpca v `dummy_data.csv`
- pre term_insurance a whole_of_life je surrender_value = 0 (čisté rizikové poistenie bez nasporenej hodnoty)
- pre endowment má zmysluplnú hodnotu
- pre annuity je síce v dátach nenulová ale v praxi sa nevypláca keďže annuity nemá lapse rates

### Diskontovanie a inflácia
- benefity aj náklady sa diskontujú pomocou diskontných faktorov odvodených z `risk_free_curve.csv`
- pre produkty `endowment` a `annuity` sa aplikuje **prémia za nelikviditu +0,5 %** (Illiquidity Premium) – diskontná sadzba sa navyšuje o 50 bázických bodov podľa princípu bottom-up (IFRS 17, odseky B72–B85); tieto produkty sú klasifikované ako `ILLIQUID_PRODUCTS` v `cashflows.py`
- pre produkty `term_insurance` a `whole_of_life` sa používa štandardná bezriziková krivka EIOPA bez úpravy
- náklady sa valorizujú infláciou:
  - base: `index_base`
  - v expense strese: percentilový geometrický blend medzi base a stressed indexom

### Paid-up a Annuity
- `paid_up`: zmluva stále aktívna a vstupuje do výpočtu, avšak peňažné príjmy (ročné poistné) sú vždy nulové – klient už zaplatil poistné vopred
- `annuity`: `annual_payment = sum_insured` (sum_insured je ročná renta), peňažné príjmy z poistného sa počítajú rovnako ako pre ostatné produkty (počas premium_payment_term, nulové pre paid_up zmluvy)

### Produkty (cash-flow)

| Produkt | Peňažné príjmy (Inflows) | Peňažné výdavky (Outflows) |
|---|---|---|
| `term_insurance` | poistné (počas premium_payment_term) | death benefit + surrender value pri lapse + náklady |
| `whole_of_life` | poistné (počas premium_payment_term) | death benefit + surrender value pri lapse + náklady |
| `endowment` | poistné (počas premium_payment_term) | death benefit + maturity benefit + surrender value pri lapse + náklady |
| `annuity` | poistné (počas premium_payment_term) | ročné dôchodkové platby + náklady |

### BEL definícia
**BEL = súčasná hodnota peňažných výdavkov − súčasná hodnota peňažných príjmov** (liability)
- kladný BEL = poisťovňa viac vyplatí ako dostane (štandardné)
- záporný BEL = portfólio je ziskové (možné pri term_insurance s vysokým poistným)

### NFR a RA (VaR)
- pre komponent `i`: `NFR_i = max(0, BEL_stress_i − BEL_base)`
- agregácia cez variance-covariance: `RA_total = sqrt(v^T * Corr * v)`
- **FCF = BEL_base + RA_total** (celkový záväzok podľa IFRS 17)

### RA po rokoch (rozpustenie cez coverage units)
- Rozpúšťanie RA je podľa IFRS 17 (odseky 44 a B119) riadené **coverage units** (jednotkami krytia) a **amortizačným faktorom**
- **Coverage units (CU)**: poistné sumy vážené podielom preživajúcich zmlúv na začiatku roka: `CU(t) = Σ (sum_insured × S_bop(t))`
- **Súčasná hodnota coverage units (PV CU)**: budúce CU diskontované **forwardovými sadzbami** – pre `endowment` a `annuity` sa použijú forwardové sadzby z krivky upravenej o prémiu za nelikviditu +0,5 %, pre ostatné produkty štandardná krivka EIOPA: `PV_CU(t) = Σ CU(s) / Π(1 + f(k))` pre s ≥ t
- **Amortizačný faktor**: `AF(t) = CU(t) / PV_CU(t)` – podiel služby poskytnutej v roku t z celkovej zostávajúcej služby
- **Rozpustenie RA**: `RA_release(t) = AF(t) × RA_BoP(t)`, kde `RA_BoP(1) = RA_total`
- **RA EoP**: `RA_EoP(t) = RA_BoP(t) − RA_release(t)` a `RA_BoP(t+1) = RA_EoP(t)`
- Súčet všetkých `RA_release` sa rovná `RA_total`
- V poslednom roku je amortizačný faktor 100 % a celý zostatok RA sa rozpustí

---

## Percentilové škálovanie šokov

Šoky sú definované na úrovni 99.5% (0.995) v `data/risk_inputs/risk_shocks_995.json`. Pre ľubovoľný percentil `p` sa škálujú cez normálny kvantil:

- `z(p) = norm.ppf(p)`
- `scale(p) = z(p) / z(0.995)`

Potom:
- multiplikatívny šok: `mult(p) = 1 + (mult(0.995) − 1) * scale(p)`
- aditívny šok: `delta(p) = delta(0.995) * scale(p)`

Pri `p = 0.60` je scale nízky → RA malé. Pri `p = 0.995` je scale = 1 → plný šok.

Metóda predpokladá že rozdelenie strát je **normálne** – šoky sa škálujú lineárne cez pomer kvantilov štandardného normálneho rozdelenia N(0,1).

---

## Rizikové komponenty

### Mortality (mortalita)
- šok: `qx * 1.15` (na 99.5%), smer nahor
- platí pre: `term_insurance`, `whole_of_life`, `endowment`

### Longevity (dlhovekosť)
- šok: `qx * 0.80` (na 99.5%), smer nadol (ľudia žijú dlhšie)
- platí pre: `annuity`, `endowment`

### Lapse (storno) – 3 scenáre, vyberie sa najhorší
- `lapse_up`: multiplikátor 1.5 na lapse rate (trvalý nárast stornovania)
- `lapse_down`: multiplikátor 0.5 s absolútnym capom 0.2 (trvalý pokles stornovania)
- `mass_lapse`: jednorazové hromadné storno 40% v prvom roku
- `NFR_lapse = max(NFR_up, NFR_down, NFR_mass)`
- platí pre: `term_insurance`, `whole_of_life`, `endowment` (annuity nemá lapse)

### Expense (náklady) – kombinovaný stres
- `expense_level`: multiplikátor 1.10 na úroveň nákladov (+10%)
- `expense_inflation`: aditívny šok +1% na infláciu nákladov
- oba šoky sa aplikujú súčasne v jednom behu
- platí pre: všetky produkty

---

## Korelačná matica

Korelačná matica je konzistentná so Solvency II štandardom:

|  | mortality | longevity | lapse | expense |
|---|---|---|---|---|
| **mortality** | 1.00 | -0.25 | 0.00 | 0.25 |
| **longevity** | -0.25 | 1.00 | 0.25 | 0.25 |
| **lapse** | 0.00 | 0.25 | 1.00 | 0.50 |
| **expense** | 0.25 | 0.25 | 0.50 | 1.00 |

`data/risk_inputs/correlation_matrix.csv`

---

## Interné testy konzistencie

Appka po výpočte zobrazí:

1. **Diskontný faktor neklesá v čase** – DF(t) ≥ DF(t+1) pre všetky roky
2. **RA neklesá s percentilom** – RA sa prepočíta pre všetky percentily a kontroluje sa monotónnosť
3. **qx po strese v [0,1]** – pravdepodobnosť úmrtia po strese musí byť v rozsahu 0 až 1
4. **lapse po strese v [0,1]** – lapse rate po strese musí byť v rozsahu 0 až 1

---

## Datové súbory – pôvod a popis

### `data/portfolio/dummy_data.csv`
Fiktívne portfólio 1000 zmlúv. Obsahuje 5 typov poistenia (`term_insurance`, `whole_of_life`, `endowment`, `annuity`, `UL_endowment`) a 4 stavy zmlúv (`new`, `paid_up`, `declined`, `closed`). Kľúčové stĺpce používané modelom:

- `insurance_type`, `agreement_state`, `date_of_birth`, `issue_date`
- `insurance_term` – doba trvania zmluvy v rokoch (9999 = doživotná pre whole_of_life)
- `premium_payment_term` – doba platenia poistného v rokoch
- `annual_premium` – ročné poistné v EUR
- `sum_insured` – poistná suma v EUR
- `surrender_value` – odkupná hodnota v EUR (0 pre term_insurance)

### `data/assumptions/mortality.csv`
Tabuľka mortality pre vek 0–104, stĺpce `age` a `qx`. Zdroj: **Štatistický úrad SR (2024)** – všeobecné populačné tabuľky mortality pre Slovensko. Obsahuje 105 riadkov (vek 0 až 104).

### `data/assumptions/lapse_rates.csv`
Ročné miery stornovania podľa produktu a doby trvania zmluvy (roky 1–50), stĺpce `product_type`, `duration_year`, `lapse_rate`. Obsahuje záznamy pre `term_insurance`, `endowment` a `whole_of_life`. `annuity` lapse rates nemá – dôchodky sa štandardne nestornujú.

### `data/assumptions/expenses.json`
Nákladové predpoklady pre každý produkt. Hodnoty boli stanovené nasledovne:
- `acquisition_rate = 1%` zo sumy – jednorazový akvizičný náklad pri uzatvorení
- `maintenance_per_policy = 25 EUR/rok` – ročné správne náklady na zmluvu
- `commission_first_year_rate = 50%` z poistného – štandardná prvá provízia
- `commission_renewal_rate = 10%` z poistného – udržiavacia provízia
- `claim_handling_per_claim = 30 EUR` – fixný náklad na likvidáciu poistnej udalosti
- pre `annuity`: `claim_handling_rate = 0.2%` z ročného dôchodku

### `data/assumptions/risk_free_curve.csv`
Bezriziková úroková krivka pre EUR, stĺpce `year`, `spot_rate`, `forward_rate`, `discount_factor`. Zdroj: **EIOPA (European Insurance and Occupational Pensions Authority)** – oficiálna mesačná publikácia RFR (Risk-Free Rate) term structures, záložka `RFR_spot_no_VA` (bez volatility adjustment) pre Slovensko/EUR, dátum **december 2025**.

Discount factor vypočítaný ako: `df(t) = 1 / (1 + spot_rate(t))^t`

Forward rate vypočítaný ako: `f(t) = (1 + spot(t))^t / (1 + spot(t-1))^(t-1) - 1`

Forward rates sa používajú pri diskontovaní coverage units pre rozpúšťanie RA.

### Prémia za nelikviditu (Illiquidity Premium)
Pre produkty `endowment` a `annuity` sa bezriziková spotová krivka navyšuje o **0,5 %** (50 bázických bodov) podľa metodológie bottom-up (IFRS 17, odseky B72–B85). Upravené krivky sa vypočítajú v `app.py` vo funkcii `_build_assumptions` pomocou konštanty `ILLIQUIDITY_PREMIUM = 0.005`:

- `spot_illiquid(t) = spot_rate(t) + 0.005`
- `discount_factors_illiquid(t) = 1 / (1 + spot_illiquid(t))^t`
- `forward_rates_illiquid(t) = (1 + spot_illiquid(t))^t / (1 + spot_illiquid(t-1))^(t-1) - 1`

Obe upravené krivky sú uložené ako samostatné polia `discount_factors_illiquid` a `forward_rates_illiquid` v objekte `Assumptions` (`cashflows.py`). Výber správnej krivky prebieha automaticky na základe `policy.insurance_type` v `project_policy_bel` (pre BEL) a v `app.py` (pre PV CU pri rozpúšťaní RA). Pre `term_insurance` a `whole_of_life` sa naďalej používa pôvodná krivka EIOPA bez úpravy.

### `data/assumptions/inflation_curve_base_and_stressed_ecb.csv`
Krivka inflácie pre roky 1–50, stĺpce `year`, `index_base`, `index_stressed`. Zdroj: **ECB (Európska centrálna banka)** – strednodobé inflačné projekcie pre eurozónu. Base scenár: inflácia 1.9–2.0% ročne smerom k cieľu 2%. Stresovaný scenár: inflácia o 1 percentuálny bod vyššia (stress addon = +1%). Index je kumulatívny (rok 1 = 1.019 pre base).

### `data/risk_inputs/risk_shocks_995.json`
Definície šokov pre všetky rizikové komponenty na úrovni spoľahlivosti 99.5%. Hodnoty sú konzistentné so **Solvency II štandardnou formulkou** (Delegated Regulation EU 2015/35):
- mortality šok: +15% na qx
- longevity šok: -20% na qx
- lapse_up: +50% na lapse rate
- lapse_down: -50% na lapse rate (s absolutným capom 20 p.b.)
- mass_lapse: 40% okamžité hromadné storno
- expense_level: +10% na náklady
- expense_inflation: +1 p.b. na infláciu nákladov

### `data/risk_inputs/correlation_matrix.csv`
Symetrická korelačná matica 4×4 pre komponenty mortality, longevity, lapse, expense. Hodnoty sú konzistentné so **Solvency II štandardnou formulkou**.

### `data/risk_inputs/product_risk_map.yml`
Mapovanie produktov na rizikové komponenty ktoré sa pre daný produkt počítajú:
- `term_insurance`: mortality, lapse, expense
- `whole_of_life`: mortality, lapse, expense
- `endowment`: mortality, longevity, lapse, expense
- `annuity`: longevity, expense

### `data/config/var_levels.json`
Zoznam povolených percentilov spoľahlivosti: `[0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.995]`.

### `data/config/var_scaling.yml`
Konfigurácia metódy škálovania šokov – normálny kvantilový pomer (normal quantile ratio) predpokladajúci normálne rozdelenie strát N(0,1). Logika implementovaná v `src/scaling.py`.

---

## Štruktúra projektu

```
project/
├── app.py                          # Hlavná Streamlit aplikácia
├── requirements.txt                # Python závislosti
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Policy dataclass, pomocné funkcie
│   ├── cashflows.py                # Projekcia BEL pre jednotlivé zmluvy
│   ├── scr.py                      # Výpočet NFR komponentov
│   ├── ra.py                       # Výpočet RA
│   ├── stresses.py                 # ShockEngine – škálovanie šokov
│   ├── scaling.py                  # VarScaler – normálny kvantil
│   ├── validation.py               # Konzistentné kontroly
│   └── io.py                       # Načítanie vstupných súborov
└── data/
    ├── portfolio/
    │   └── dummy_data.csv
    ├── assumptions/
    │   ├── mortality.csv
    │   ├── lapse_rates.csv
    │   ├── expenses.json
    │   ├── risk_free_curve.csv
    │   └── inflation_curve_base_and_stressed_ecb.csv
    ├── risk_inputs/
    │   ├── risk_shocks_995.json
    │   ├── correlation_matrix.csv
    │   └── product_risk_map.yml
    └── config/
        ├── var_levels.json
        └── var_scaling.yml
```

---

## Spustenie lokálne

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Závislosti (`requirements.txt`)
```
streamlit>=1.33
pandas>=2.0
numpy>=1.24
scipy>=1.10
pyyaml>=6.0
matplotlib>=3.7
```
