# StockResearchMarket

Środowisko do badania strategii inwestycyjnych na akcjach i ETF-ach: długie backtesty 10-20 lat, wiele źródeł danych, automatyczne szukanie parametrów, walidacja walk-forward i raporty HTML z wykresami.

Projekt jest wzorowany na `researchmarket`, ale startuje jako czysty research lab dla rynku akcji: bez frontendu i bez live tradingu, za to z działającym CLI, cache danych i powtarzalnymi eksperymentami.

## Szybki Start

```bash
make setup
make smoke
```

Smoke test działa offline na syntetycznym panelu cenowym i zapisuje pierwszy raport w `experiments/`.

Backtest 20 lat dla SPY:

```bash
uv run stockresearch run --strategy sma_cross --tickers SPY --years 20
```

Optymalizacja parametrów:

```bash
uv run stockresearch optimize --strategy sma_cross --ticker SPY --method grid --years 20
```

Portfelowa rotacja momentum na domyślnym koszyku ETF/akcji:

```bash
uv run stockresearch portfolio --years 20
```

GARP-like multifactor baseline:

```bash
uv run stockresearch garp-run --experiment 001_baseline_garp --provider synthetic --years 10
```

SEC EDGAR fundamentals + yfinance prices smoke:

```bash
SEC_USER_AGENT="StockResearchMarket/0.1 your-email@example.com" \
uv run stockresearch garp-run --experiment sec_yfinance_smoke --years 3
```

Aktualny skład S&P 500 cofnięty 10 lat, price-only tests:

```bash
uv run python scripts/run_current_sp500_price_tests.py
```

Autoresearch wariantów GARP:

```bash
uv run stockresearch garp-autoresearch --provider synthetic --years 8 --max-experiments 6
```

## Co Jest W Środku

- `stockresearchmarket/data/` - pobieranie i cache danych z `yfinance`, `stooq`, CSV oraz tryb syntetyczny.
- `stockresearchmarket/strategies/` - strategie bazowe: buy and hold, SMA cross, RSI mean reversion, Donchian breakout, dual momentum.
- `stockresearchmarket/engine/` - wektorowy silnik backtestów jednoaktywnych i portfelowych z kosztami, poślizgiem oraz benchmarkiem.
- `stockresearchmarket/optimization/` - grid search, opcjonalnie Optuna, oraz walk-forward.
- `stockresearchmarket/reports/` - raporty HTML/CSV z equity curve, drawdownem, rolling Sharpe i miesięcznymi stopami zwrotu.
- `stockresearchmarket/garp/` - GARP-like multifactor ranking, universe builder, scoring, rebalancing, reports, autoresearch.
- `configs/default.yaml` - domyślny zakres 20 lat, koszyk instrumentów, koszty i siatki parametrów.
- `configs/garp_default.yaml` i `configs/garp_experiments/` - konfiguracje GARP 001-006.

## Źródła Danych

Domyślnie projekt używa `yfinance`, bo jest szybki do researchu i nie wymaga kluczy API. Dostępne są też:

- `stooq` przez `pandas-datareader`;
- `csv` z plików `data/raw/<TICKER>.csv`;
- `synthetic` do testów offline;
- `sec_edgar` dla point-in-time fundamentals z SEC `companyfacts` z datą publikacji filingów jako `as_of_date`;
- miejsce na premium API: Polygon, Nasdaq Data Link, Alpha Vantage.

Wyniki pobrań są cache'owane jako Parquet w `data/historical/`.
Fundamentals SEC są cache'owane w `data/fundamentals/`; przed pobieraniem ustaw `SEC_USER_AGENT` z nazwą aplikacji i kontaktem.

## Ważne Założenia

To jest narzędzie badawcze, nie rekomendacja inwestycyjna. Każdy wynik trzeba czytać z porównaniem do buy and hold, kosztami transakcyjnymi, stabilnością walk-forward i ryzykiem przeoptymalizowania.
