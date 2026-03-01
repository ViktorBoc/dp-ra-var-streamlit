# Risk Adjustment (VaR) pre životné poistenie (Streamlit)

Tento projekt je webová aplikácia v **Streamlit**, ktorá pre vybrané portfólio životných zmlúv vypočíta **Risk Adjustment (RA)** metódou **VaR** podľa štandardu **IFRS 17**. Rizikové komponenty sa agregujú cez **variance–covariance** prístup (korelačná matica). Všetko funguje iba z lokálnych súborov (žiadne externé DB ani internetové API).

---

## Čo appka počíta (výstupy)

Používateľ vyberie:
- `insurance_type` (typ poistenia / produktu)
- percentil `p` z `{0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.995}` (načítané z `data/config/var_levels.json`)

Appka vypočíta pre portfólio všetkých zmlúv daného typu (len aktívne stavy):

- **BEL základný (EUR)** = PV(outflows) − PV(inflows)
- **Riziková prirážka RA (EUR)** = agregované NFR komponenty cez korelačnú maticu
- **Sadzba RA (%)** = RA_total / BEL_base
- **Peňažné toky na splnenie záväzkov – FCF (EUR)** = BEL_base + RA_total (IFRS 17 záväzok)
- **NFR komponenty (EUR)**: `mortality`, `longevity`, `lapse`, `expense` vrátane BEL šokovaného pre každý komponent
- **RA po rokoch** = tabuľka rozpustenia RA počas trvania portfólia
- **Grafy**: BEL a RA po rokoch, NFR komponenty, RA release, BEL základný vs šokovaný
- **Kontrola konzistencie** (OK/CHYBA)

Export (CSV s UTF-8 BOM kódovaním pre správne zobrazenie v Exceli):
- `ra_results.csv` – súhrnné výsledky RA
- `nfr_components.csv` – NFR komponenty s BEL šokovaným
- `ra_po_rokoch.csv` – rozpustenie RA po rokoch
- Grafy stiahnuteľné ako PNG

---

## Kľúčové pravidlá

### Portfólio (filter)
- zahrnuté len `agreement_state ∈ {new, paid_up}`
- `declined`, `closed` ignorované
- `UL_endowment` je vylúčený z UI (finančné riziká nie sú modelované)

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
- inflow (poistné) sa zastaví po uplynutí `remaining_ppt` rokov projekcie

### Surrender Value (odkupná hodnota)
- pri každom storne (lapse) sa vypláca odkupná hodnota klientovi
- implementovaná ako **lineárna interpolácia klesajúca k nule**:
  - `sv(t) = surrender_value * max(0, horizon - t) / horizon`
  - v roku 0 = plná odkupná hodnota, v poslednom roku = 0
- hodnota `surrender_value` sa načítava priamo zo stĺpca v `dummy_data.csv`
- pre term_insurance je typicky 0 (čisté rizikové poistenie bez nasporenej hodnoty)

### Diskontovanie a inflácia
- benefity aj náklady sa diskontujú pomocou diskontných faktorov z `risk_free_curve.csv`
- náklady sa valorizujú infláciou:
  - base: `index_base`
  - v expense strese: percentilový geometrický blend medzi base a stressed indexom

### Paid-up a Annuity
- `paid_up`: premium inflow = 0 (zmluva je splatená, klient už neplatí)
- `annuity`: `annual_payment = sum_insured` (sum_insured je ročná renta), premium inflow = 0

### Produkty (cash-flow)

| Produkt | Inflows | Outflows |
|---|---|---|
| `term_insurance` | poistné (počas premium_payment_term) | death benefit + surrender value pri lapse + náklady |
| `whole_of_life` | poistné (počas premium_payment_term) | death benefit + surrender value pri lapse + náklady |
| `endowment` | poistné (počas premium_payment_term) | death benefit + maturity benefit + surrender value pri lapse + náklady |
| `annuity` | žiadne | ročné dôchodkové platby + náklady |

### BEL definícia
- **BEL = PV(outflows) − PV(inflows)** (liability)
- kladný BEL = poisťovňa viac vyplatí ako dostane (štandardné)
- záporný BEL = portfólio je ziskové (možné pri term_insurance s vysokým poistným)

### NFR a RA (VaR)
- pre komponent `i`: `NFR_i = max(0, BEL_stress_i − BEL_base)`
- agregácia cez variance-covariance: `RA_total = sqrt(v^T * Corr * v)`
- `RA_rate = RA_total / BEL_base`
- **FCF = BEL_base + RA_total** (celkový záväzok podľa IFRS 17)

### RA po rokoch (rozpustenie)
- `RA_rate` je **konštantná** – vypočítaná raz pri rok 0 (initial recognition)
- `BEL_rok_t` sa prepočítava každý rok posunutím horizontu portfólia o t rokov
- `RA_rok_t = RA_rate * BEL_rok_t` – RA klesá spolu s BEL
- `RA_release_t = RA_rok_{t-1} - RA_rok_t` – ročné rozpustenie RA do výsledovky
- súčet všetkých RA_release sa rovná RA_total v roku 0

---

## Percentilové škálovanie šokov

Šoky sú definované na úrovni 99.5% (0.995) v `data/risk_inputs/risk_shocks_995.json`. Pre ľubovoľný percentil `p` sa škálujú cez normálny kvantil:

- `z(p) = norm.ppf(p)`
- `scale(p) = z(p) / z(0.995)`

Potom:
- multiplikatívny šok: `mult(p) = 1 + (mult(0.995) − 1) * scale(p)`
- aditívny šok: `delta(p) = delta(0.995) * scale(p)`

Pri `p = 0.60` je scale nízky → RA malé. Pri `p = 0.995` je scale = 1 → plný šok.

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
- platí pre: `term_insurance`, `whole_of_life`, `endowment`

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

Zdroj: `data/risk_inputs/correlation_matrix.csv` – hodnoty určené na základe Solvency II štandardnej formulky.

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
Fiktívne portfólio 1000 životných zmlúv. Obsahuje 5 typov poistenia (`term_insurance`, `whole_of_life`, `endowment`, `annuity`, `UL_endowment`) a 4 stavy zmlúv (`new`, `paid_up`, `declined`, `closed`). Dáta boli generované náhodne ako demo portfólio. Kľúčové stĺpce používané modelom:

- `insurance_type`, `agreement_state`, `date_of_birth`, `issue_date`
- `insurance_term` – doba trvania zmluvy v rokoch (9999 = doživotná pre whole_of_life)
- `premium_payment_term` – doba platenia poistného v rokoch
- `annual_premium` – ročné poistné v EUR
- `sum_insured` – poistná suma v EUR
- `surrender_value` – odkupná hodnota v EUR (0 pre term_insurance)

**Úprava poistného pre term_insurance:** keďže náhodne generované poistné bolo príliš vysoké voči riziku (priemer ~3.4% zo sumy), bolo prenasobené koeficientom 0.15 čím sa priemer znížil na ~0.5% zo sumy – realistickejší pomer pre rizikové životné poistenie.

### `data/assumptions/mortality.csv`
Tabuľka mortality pre vek 0–104, stĺpce `age` a `qx`. Zdroj: **Štatistický úrad SR** – všeobecné populačné tabuľky mortality pre Slovensko. Obsahuje 105 riadkov (vek 0 až 104).

### `data/assumptions/lapse_rates.csv`
Ročné miery stornovania podľa produktu a doby trvania zmluvy (roky 1–50), stĺpce `product_type`, `duration_year`, `lapse_rate`. Hodnoty boli stanovené odborne – realistické lapse rates pre slovenský trh:
- rok 1: 12% (term), klesá do roka 10 na 6% a stabilizuje sa
- `annuity` nemá lapse (dôchodky sa štandardne nestornujú)
- `UL_endowment` nie je zahrnutý

### `data/assumptions/expenses.json`
Nákladové predpoklady pre každý produkt. Hodnoty boli odborne stanovené:
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
Konfigurácia metódy škálovania šokov (normal quantile ratio). Dokumentačný súbor, logika je implementovaná v `src/scaling.py`.

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
