# Risk Adjustment (VaR) pre životné poistenie (Streamlit)

Tento projekt je jednoduchá webová aplikácia v **Streamlit**, ktorá pre vybrané portfólio životných zmlúv vypočíta **Risk Adjustment (RA)** metódou **VaR**. Rizikové komponenty sa agregujú cez **variance–covariance** prístup (korelačná matica). Všetko funguje iba z lokálnych súborov (žiadne externé DB ani internetové API).

---

## Čo appka počíta (výstupy)

Používateľ vyberie:
- `insurance_type` (typ poistenia / produktu)
- percentil `p` z `{0.50, 0.55, …, 0.95, 0.995}` (načítané z `data/config/var_levels.json`)

Appka vypočíta pre portfólio všetkých zmlúv daného typu (len aktívne stavy):
- **BEL_base (EUR)** = PV(outflows) − PV(inflows)
- **SCR komponenty (EUR)**: `mortality`, `longevity`, `lapse`, `expense`
- **SCR_total / RA_total (EUR)** = agregácia cez koreláciu
- **RA_rate (%)** = RA_total / BEL_base (ošetrené delenie nulou)
- **Consistency checks** (PASS/FAIL)

Export:
- **Download CSV** (funguje aj po deployi – stiahne súbor používateľovi)
- **Save to outputs/** (funguje lokálne; pri deployi ukladá na server – skôr debug)

---

## Kľúčové pravidlá

### Portfólio (filter)
- zahrnuté len `agreement_state ∈ {new, paid_up}`
- `declined`, `closed` ignorované

### Valuation date a projekčný horizont
- valuation date je fixný: **2026-01-01**
- výpočet veku a trvania:
  - `current_age = floor((valuation_date - date_of_birth).days / 365.25)`
  - `duration = floor((valuation_date - issue_date).days / 365.25)`
  - `remaining_term = max(0, insurance_term - duration)`
  - ak `insurance_term == 9999` → berie sa ako veľmi dlhé, ale aj tak sa zreže horizontom
- `max_age = 105`
- horizont na zmluvu:
  - `T_i = min(remaining_term, 50, max_age - current_age)`

### Diskontovanie a inflácia
- benefity aj náklady sa diskontujú pomocou `risk_free_curve.csv` (discount_factor)
- náklady sa valorizujú infláciou:
  - base: `index_base`
  - v expense strese: používa sa stresovaný index (resp. percentilový blend medzi base a stressed)

### Paid-up a Annuity
- paid_up: premium inflow = 0
- annuity:
  - `annual_payment = sum_insured` (sum_insured je ročná renta)
  - premium inflow = 0

### Produkty (cash-flow)
- `term_insurance`: premium inflow + death benefit outflow + expenses
- `whole_of_life`: premium inflow + death benefit outflow + expenses (horizont max 50 / max_age)
- `endowment`: premium inflow + death benefit outflow + maturity benefit v poslednom roku + expenses
- `UL_endowment`: pre MVP modelované rovnako ako endowment (finančné riziká ignorované)
- `annuity`: ročné platby + expenses

### BEL definícia
- **BEL = PV(outflows) − PV(inflows)** (liability)

### SCR a RA (VaR)
- pre komponent `i`:
  - `SCR_i = max(0, BEL_stress_i − BEL_base)`
- agregácia:
  - `SCR_total = sqrt(v^T Corr v)`
  - `RA_total = SCR_total`
- `RA_rate = RA_total / BEL_base` (safe-div)

---

## Percentilové škálovanie šokov

Šoky sú definované na úrovni 99.5% (0.995) v `data/risk_inputs/risk_shocks_995.json`. Pre ľubovoľný percentil `p` sa škálujú cez normálny kvantil:

- `z(p) = norm.ppf(p)`
- `scale(p) = z(p) / z(0.995)`

Potom:
- multiplikatívny šok:
  - `mult(p) = 1 + (mult(0.995) − 1) * scale(p)`
- aditívny šok:
  - `delta(p) = delta(0.995) * scale(p)`

Pri `p = 0.50` je `z(0.50) = 0` ⇒ scale = 0 ⇒ stres ~0 ⇒ RA ~0.

---

## Rizikové komponenty (ako sa stresuje)

### Mortality
- stresuje sa `qx` multiplikátorom (clamp na [0,1])

### Longevity
- stresuje sa `qx` multiplikátorom (clamp na [0,1])

### Lapse (3 scenáre, najhorší)
Urobia sa 3 behy:
- `lapse_up`
- `lapse_down`
- `mass_lapse`

Pre každý:
- `SCR_scenario = max(0, BEL_stress − BEL_base)`

A finálne:
- `SCR_lapse = max(SCR_up, SCR_down, SCR_mass)`

### Expense (kombinovaný stres)
- v jednom behu sa aplikuje:
  - `expense_level` multiplikátor na úroveň nákladov
  - `expense_inflation`: stresovaný index (resp. percentilový blend)
- `SCR_expense = max(0, BEL_expense_stress − BEL_base)`

---

## Interné testy konzistencie (PASS/FAIL)

Appka po výpočte zobrazí:
1. **discount_factor non-increasing by year**  
   Diskontné faktory musia byť neklesajúce v čase (DF_t ≥ DF_{t+1}).

2. **p=0.50 ⇒ RA approx 0**  
   Pri mediáne je scale=0 ⇒ stres ~0 ⇒ RA ~0.

3. **RA non-decreasing with percentile**  
   RA sa prepočíta pre všetky percentily z `var_levels.json` a kontroluje sa monotónnosť.

4. **qx after stress in [0,1]**  
   qx po strese musí zostať pravdepodobnosť.

5. **lapse after stress in [0,1]**  
   lapse po strese musí zostať pravdepodobnosť.

---

## Súbory a čo obsahujú

### `data/portfolio/dummy_data.csv`
Portfólio zmlúv. Očakávané stĺpce:
- `insurance_type`
- `agreement_state`
- `date_of_birth` (YYYY-MM-DD)
- `issue_date` (YYYY-MM-DD)
- `insurance_term` (roky; 9999 = veľmi dlhé)
- `sum_insured`
- `premium` (môže chýbať)

**Fallback premium:** ak chýba/NaN, použije sa `2% * sum_insured`.

### `data/assumptions/mortality.csv`
- `age`, `qx`

### `data/assumptions/lapse_rates.csv`
- `product_type`
- `duration_year` (1..50)
- `lapse_rate`

### `data/assumptions/expenses.json`
Globálne náklady (rovnaké pre všetky produkty).  
Používané kľúče (ak existujú):
- `acquisition_per_policy`
- `maintenance_per_policy`
- `commission_first_year_rate`
- `commission_renewal_rate`
- `claim_handling_per_claim`  
Chýbajúce kľúče sa berú ako 0.

### `data/assumptions/risk_free_curve.csv`
- `year` (1..50)
- `discount_factor` (diskontovanie cash-flow)

### `data/assumptions/inflation_curve_base_and_stressed_ecb.csv`
- `year` (1..50)
- `index_base`
- `index_stressed`

### `data/risk_inputs/risk_shocks_995.json`
Definície šokov na 0.995 + typ šoku (mult/add/…).

### `data/risk_inputs/correlation_matrix.csv`
Korelačná matica pre komponenty:
- mortality, longevity, lapse, expense

### `data/risk_inputs/product_risk_map.yml`
Mapovanie produktu na komponenty, ktoré sa majú počítať.

### `data/config/var_levels.json`
Zoznam povolených percentilov (UI + validácia monotónnosti).

### `data/config/var_scaling.yml`
Konfig súbor (momentálne nepoužitý v logike; ponechaný kvôli štruktúre).

---

## Spustenie lokálne

### 1) Vytvor venv (odporúčané)
Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py