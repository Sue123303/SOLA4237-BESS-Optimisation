# Household BESS Optimisation – SOLA 4237 Assignment 2

Battery energy storage (BESS) optimisation for a single residential site at Mt Buller, VIC, under four services:

1. a flat-rate tariff
2. a time-of-use (ToU) tariff
3. wholesale pass-through pricing (Amber-style, with a $25/month VPP fee)
4. wholesale pass-through under PV and price uncertainty (48-hour rolling horizon, hourly re-plan, 5-minute execution)

All analysis is in one Jupyter notebook: [`BESS_Optimisation.ipynb`](BESS_Optimisation.ipynb).

## Key results

Results are for the selected 5 kWh / 2.5 kW battery (BYD Battery-Box Premium HVS 5.1 with an SMA Sunny Boy Storage 2.5 AC-coupled inverter). Bills include energy charges and the VPP fee where it applies; the daily standing charge is excluded.

| Tariff (year, foresight) | Bill, no BESS | Bill, self-consumption rule | Bill, optimised | Payback with rebate |
| --- | --- | --- | --- | --- |
| Flat (2023, perfect) | $180 | −$73 | −$73 | 18.5 yr |
| ToU (2023, perfect) | $101 | −$127 | −$162 | 17.8 yr |
| Wholesale + VPP (2023, perfect) | $545 | $291 | $133 | 11.4 yr |
| Wholesale + VPP (2024, forecasts) | $460 | $189 | $146 | 14.9 yr |

- No battery size has a positive 10-year NPV, even with the federal rebate.
- An AC-coupled retrofit is used: the existing PV inverter is kept, so no hybrid inverter is needed.
- With forecasts the battery captures 49% of the perfect-foresight saving. Price forecast error costs $319/yr; PV forecast error costs $3/yr.
- With forecasts, VPP optimisation is worth $343/yr; after the $300/yr fee the customer nets $43/yr (break-even fee $28.6/month).

## Notebook structure

| Section | Content |
| --- | --- |
| **Step 1 – Telemetry cleaning and augmenting** | |
| 1.1 | Loading data, handling of time (interval-ending AEST) and resampling |
| 1.2 | Missing data: identification and cleaning |
| 1.3 | Suspicious / spurious data checks |
| 1.4 | Site load, solar and weather analysis |
| 1.5 | Power vs energy calculations |
| 1.6 | 2024 wholesale prices from AEMO |
| 1.7 | AusNet NAST11 network tariff and customer price signals |
| **Step 2 – BESS analysis and optimisation** | |
| 2.1 | Battery model, AC-coupled topology and losses, operational constraints |
| 2.2 | Optimisation objective function and constraints (LP) |
| 2.3 | BESS sizing from load vs generation and economics; matching commercial product |
| 2.4.1–2.4.3 | Operation and optimisation under flat, ToU and wholesale tariffs |
| 2.5.1 | PV forecast model (trained on 2023, forecast 2024, compared with the commercial forecast: MAE, RMSE, nRMSE) |
| 2.5.2 | Wholesale price forecast |
| 2.5.3 | 48-hour rolling-horizon scheduler with 5-minute execution |
| 2.5.4 | Forecasts vs perfect foresight; PV vs price uncertainty |
| **Step 3 – Metrics** | |
| 3.1 | Annual bill, self-consumption, self-sufficiency |
| 3.2 | Savings against no BESS and against the self-consumption rule |
| 3.3 | Payback and techno-economic analysis (incl. break-even battery cost) |
| 3.4 | Value of and cost to the customer of VPP optimisation |
| 3.5 | Revenue analysis and bill breakdown across tariffs |

The notebook contains section headings, code and outputs only; explanations are in the accompanying report. Code comments of the form `# Report Table n` / `# Report Figure n` mark where each report table and figure is produced.

## Method in brief

- **Optimiser**: linear program solved with HiGHS via `scipy.optimize.linprog`. Objective = import cost − export revenue + degradation cost ($0.04 per kWh of throughput). Constraints: power balance, SoC dynamics (η = 0.946 each way, 89.4% round trip), SoC 10–95%, cold-temperature charge derating, PV curtailment (zero export at negative prices).
- **Benchmark**: rule-based self-consumption (charge from surplus PV only, discharge to meet load).
- **Perfect foresight**: full year solved in 7-day windows with 2-day look-ahead.
- **Uncertainty**: hourly 48-hour plan using forecast PV (gradient-boosting model on weather) and forecast price (blend of persistence and a 14-day time-of-day profile); 5-minute execution; settlement on actual price, PV and load. Load is assumed perfectly known, as the brief allows.
- **Wholesale customer price**: spot × 1.05 loss factor + AusNet NAST11 network charge + certificates, market fees and hedging + GST; export = loss-adjusted spot.
- Every optimisation result is checked independently: power balance, SoC reconstruction, SoC limits, charge limits, and no simultaneous charge/discharge or import/export.

## Data

| Path | Content | Source |
| --- | --- | --- |
| `Data_VIC_Mt Buller - Alpine_2023-2024_partial.csv` | 5-minute load, PV, weather, tariffs; 2023 spot prices; 2024 commercial PV forecast | Course data |
| `data/aemo/` | VIC1 5-minute RRP, Dec 2023 – Dec 2024 (and Jan 2023 for validation) | AEMO Aggregated Price and Demand files (`https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_YYYYMM_VIC1.csv`) |
| `data/tariffs/` | AusNet Services schedules of tariffs, FY2022-23 to FY2024-25 | AusNet Services / AER |
| `data/cache/` | Cached rolling-horizon (MPC) results | Generated by the notebook |

Missing AEMO files are downloaded automatically when the notebook runs; this needs internet access.

## How to run

Requirements (tested versions): Python 3.13, pandas 2.2, numpy 2.1, scipy 1.15, scikit-learn 1.6, matplotlib 3.10, nbformat 5.10, Jupyter.

```bash
pip install pandas numpy scipy scikit-learn matplotlib nbformat jupyter

# run the notebook end to end
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 BESS_Optimisation.ipynb
```

Or open `BESS_Optimisation.ipynb` in Jupyter and run all cells. Run from the repository root so the relative data paths resolve.

- **Runtime**: about 5 minutes with the cache present. Without `data/cache/`, the four rolling-horizon scenarios (A–D) add several minutes (about 8,800 LP solves each).
- **Cache**: file names include the battery size, degradation cost, forecast settings and tariff version. Delete `data/cache/` to force a full recompute after changing anything else that affects the MPC runs.

## Regenerating the notebook

The notebook is generated from [`build_notebook.py`](build_notebook.py). Edit that script, then run:

```bash
python build_notebook.py   # writes BESS_Optimisation.ipynb (without outputs)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 BESS_Optimisation.ipynb
```

## Main assumptions

| Item | Value | Basis |
| --- | --- | --- |
| Product | BYD HVS 5.1 + SMA Sunny Boy Storage 2.5 | Modelled as 5.0 kWh nameplate (datasheet: 5.12 kWh usable) |
| Topology | AC-coupled retrofit | Measured PV is AC output of the existing inverter; a DC hybrid would need a new hybrid inverter |
| Efficiency | 97% inverter × 97.5% cells, each way | SBS 2.5: 97% max efficiency; BYD HVS: ≥96% DC round trip |
| Degradation cost | $0.04/kWh throughput (each of charge and discharge) | $350/kWh cells ÷ (6,000 cycles × 0.85 DoD) ≈ $0.069 per kWh cycled (≈ $0.034 each way), rounded up as a conservative allowance |
| Battery cost | $2,000 + $850/kWh installed | Market estimate |
| Rebate | $370 per usable kWh | Cheaper Home Batteries estimate |
| Economics | 10-year life, 5% discount rate | Assumption |
| Network tariff | AusNet NAST11, peak 15:00–21:00 local time | Official AusNet schedules |
| Certificates, market fees, hedging, loss factor | 1.6 c, 0.3 c, 1.0 c/kWh; × 1.05 | Assumptions |

## Limitations

- Only the AC-coupled retrofit topology is modelled.
- Temperature derating covers cold-weather charging only.
- Degradation cost is linear in throughput; depth of discharge and calendar ageing are not modelled.
- The 2024 "actual" PV is the commercial forecast, and the PV forecast uses observed weather, so PV forecast error is optimistic.
- Beyond the first 2 hours, the 48-hour plan uses 30-minute resolution for speed.
