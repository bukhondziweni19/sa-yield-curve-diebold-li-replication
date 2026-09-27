# Forecasting the Term Structure of South African Government Bond Yields

A replication of **Diebold, F.X. and Li, C. (2006) 'Forecasting the term structure of government bond yields', *Journal of Econometrics*, 130(2), pp. 337–364**, using South African government bond and Treasury bill yields.

Empirical Replication Project for **DDM8X01 Debt Market Modelling**, Master of Financial Engineering, University of Johannesburg (2026).

**Authors:** Bukho Ndziweni and Sizwe November

---

## Overview

Diebold and Li (2006) reinterpret the Nelson-Siegel yield curve as a dynamic three-factor model of level, slope and curvature, model each factor as a first-order autoregressive (AR(1)) process, and show that the resulting forecasts beat standard benchmarks at a 12-month horizon for US Treasury yields (1985 to 2000).

This project applies the same method to South Africa:

1. **Factor estimation:** the Nelson-Siegel curve, with the decay parameter fixed at λ = 0.0609, is fitted by ordinary least squares (OLS) to monthly yields at five maturities (3, 18, 48, 90 and 150 months), January 1994 to February 2023.
2. **Factor dynamics:** each factor is modelled as an AR(1), with augmented Dickey-Fuller (ADF) unit root tests and residual diagnostics.
3. **Out-of-sample forecasting:** recursive, direct forecasts at horizons of 1, 6 and 12 months from January 2003, compared with a random walk, AR(1) and VAR(1) (vector autoregression) models on yields, Nelson-Siegel with VAR(1) factors, and the slope regression, using root mean squared errors (RMSE) and Diebold-Mariano tests.
4. **Diagnostics and robustness:** a Nelson-Siegel random walk, rolling-window estimation, sub-period evaluation, an alternative maturity for the longest bond band, and a fit using all available maturities to May 2026.

## Key findings

- **The factor structure replicates.** The estimated factors correlate at 0.979, −0.995 and 0.941 with the empirical level, slope and curvature, and share the persistence and unit root properties reported for the US.
- **The forecasting result does not.** Over 2003 to 2023, a random walk forecasts South African yields more accurately than the Nelson-Siegel model at all three horizons, significantly so at most maturities.
- **Two identifiable causes:**
  - at short horizons, fitting error in banded, mixed-instrument yield data;
  - at long horizons, mean reversion towards a historical average that spans monetary policy regime changes (the 1998 crisis and the move to inflation targeting in 2000).
- **Partial recovery within a single regime.** With a rolling window that excludes the pre-inflation-targeting period, the model beats the random walk at 48 and 90 months in 2013 to 2023.

## Repository structure

```
├── Code/
│   ├── 00_download_tbills.ipynb    Downloads SARB Treasury bill rates (run once; optional)
│   ├── 01_replication.ipynb        Main analysis, Steps 1 to 11
│   └── 01_replication.py           Script version of the main notebook
├── Data/
│   └── Raw/
│       ├── SARB_govt_bond_yields_monthly.xlsx
│       ├── SARB_treasury_bills_daily.csv
│       └── Data_Sources.txt        Series codes, descriptions and download date
├── Output/
│   ├── 01_replication_output.html  Complete software output
│   ├── 01_replication_output.pdf
│   ├── 00_download_tbills_output.html
│   ├── sa_yields_monthly_clean.csv
│   ├── ns_factors_main.csv
│   ├── ns_factors_all_maturities.csv
│   └── fig1 to fig8 (.png)
├── Report/                          Empirical report (PDF)
├── ReadMe.pdf                       Submission ReadMe
├── requirements.txt
└── README.md
```

## Data

All data are from the South African Reserve Bank (SARB), downloaded on 27 September 2026.

| Series | Codes | Frequency | Source |
|---|---|---|---|
| Government bond yields by maturity band (0 to 3, 3 to 5, 5 to 10, 10+ and 20 to 30 years) | KBP2000M, KBP2001M, KBP2002M, KBP2003M, KBP2049M | Monthly average | SARB Online Statistical Query |
| Treasury bill tender rates (91, 182, 273 and 364 days) | MMRD203A, MMRD206A, MMRD209A, MMRD212A | Daily | SARB web data service |

Raw files are stored exactly as downloaded; all cleaning is done in code. Treasury bill discount rates are converted to yields, daily rates are averaged to monthly, and zero values in the 3 to 5 year series (March 2023 to January 2025) are treated as missing.

## How to reproduce

1. Install Python 3 (for example, via Anaconda) and the packages in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
2. Keep the folder structure unchanged. The code uses relative paths and must be run from inside the `Code` folder.
3. Open `Code/01_replication.ipynb` in Jupyter and select **Kernel → Restart & Run All**. The notebook runs from start to finish in under a minute.

   Alternatively, from the `Code` folder: `set MPLBACKEND=Agg` (Windows) or `export MPLBACKEND=Agg` (Mac or Linux), then `python 01_replication.py`.

`00_download_tbills.ipynb` does not need to be rerun. The SARB revises and extends its data, so a new download may differ slightly from the file used here.

## Software

Python 3.13.9, pandas 2.3.3, NumPy 2.3.5, statsmodels 0.14.5, Matplotlib 3.10.6, SciPy 1.16.3.

## Reference

Diebold, F.X. and Li, C. (2006) 'Forecasting the term structure of government bond yields', *Journal of Econometrics*, 130(2), pp. 337–364.
