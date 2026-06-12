# GARP Data Limitations

This project separates research plumbing from investable evidence.

## Current MVP

- Synthetic fundamentals and estimates are point-in-time safe fixtures for framework validation.
- SEC EDGAR `companyfacts` fundamentals are parsed with SEC filing dates as `as_of_date`.
- The SEC adapter computes valuation ratios with yfinance prices observed on or before each filing date.
- yfinance/Stooq daily prices are useful for price research, but not survivorship-free universe research.
- yfinance current fundamentals are not point-in-time safe and must not be used in the main GARP backtest.
- Analyst estimates are only used when the provider supplies an `as_of_date` or revision date.
- yfinance analyst/fundamental snapshots are not used as revisions because they are not historical point-in-time data.

## Bias Controls

- Every factor row must be keyed by `as_of_date`.
- A rebalance can use only rows where `as_of_date <= rebalance_date`.
- Factors without safe historical availability are marked `unavailable` or `unsafe`.
- Missing revisions automatically redistribute their weight over available growth, quality, value, and momentum categories.
- Universe membership from today's S&P 500 list is not accepted as survivorship-bias-free evidence.

## Production Data To Add

- SEC EDGAR company facts for official US statements.
- Financial Modeling Prep / SimFin for normalized statements and ratios.
- Finnhub / FMP / Alpha Vantage for analyst estimates when revision dates are available.
- CRSP, Norgate, or S&P Global for point-in-time historical index constituents.
