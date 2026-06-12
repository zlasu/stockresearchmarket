# Open GARP Multifactor Backtester

The GARP module is an iterative research framework for US large/mid cap equity ranking. It supports:

- point-in-time factor snapshots via `as_of_date`;
- growth, value, quality, momentum, revisions, and risk factors;
- cross-sectional percentile scoring with winsorization;
- automatic weight redistribution when revisions or fundamentals are unavailable;
- monthly or quarterly rebalancing;
- equal-weight, inverse-volatility, and MVP HRP/ERC approximation;
- SMA200 market filter, sector momentum, turnover reduction, and volatility targeting experiments;
- HTML/Markdown tear sheets and autoresearch leaderboards.

## Commands

```bash
uv run stockresearch garp-run --experiment 001_baseline_garp --provider synthetic --years 10
uv run stockresearch garp-run --experiment 002_garp_sma200_filter --provider synthetic --years 10
uv run stockresearch garp-autoresearch --provider synthetic --years 8 --max-experiments 6
```

Real-data smoke with SEC EDGAR fundamentals and yfinance prices:

```bash
SEC_USER_AGENT="StockResearchMarket/0.1 your-email@example.com" \
uv run stockresearch garp-run --experiment sec_yfinance_smoke --years 3 --refresh
```

`sec_edgar` uses SEC `companyfacts` and filing dates as `as_of_date`. yfinance supplies daily prices, benchmark prices, market-cap price inputs, and optional sector metadata. Analyst revisions are disabled in this config because yfinance estimate snapshots are not historical point-in-time data.

## Current Data Position

Synthetic fundamentals and estimates are used for framework validation. Real yfinance fundamentals are intentionally not used in the main GARP backtest because they are not historical point-in-time snapshots. Add real providers only when they can supply `as_of_date` or equivalent filing/revision dates.
