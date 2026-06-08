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

## Co Jest W Środku

- `stockresearchmarket/data/` - pobieranie i cache danych z `yfinance`, `stooq`, CSV oraz tryb syntetyczny.
- `stockresearchmarket/strategies/` - strategie bazowe: buy and hold, SMA cross, RSI mean reversion, Donchian breakout, dual momentum.
- `stockresearchmarket/engine/` - wektorowy silnik backtestów jednoaktywnych i portfelowych z kosztami, poślizgiem oraz benchmarkiem.
- `stockresearchmarket/optimization/` - grid search, opcjonalnie Optuna, oraz walk-forward.
- `stockresearchmarket/reports/` - raporty HTML/CSV z equity curve, drawdownem, rolling Sharpe i miesięcznymi stopami zwrotu.
- `configs/default.yaml` - domyślny zakres 20 lat, koszyk instrumentów, koszty i siatki parametrów.

## Źródła Danych

Domyślnie projekt używa `yfinance`, bo jest szybki do researchu i nie wymaga kluczy API. Dostępne są też:

- `stooq` przez `pandas-datareader`;
- `csv` z plików `data/raw/<TICKER>.csv`;
- `synthetic` do testów offline;
- miejsce na premium API: Polygon, Nasdaq Data Link, Alpha Vantage.

Wyniki pobrań są cache'owane jako Parquet w `data/historical/`.

## Ważne Założenia

To jest narzędzie badawcze, nie rekomendacja inwestycyjna. Każdy wynik trzeba czytać z porównaniem do buy and hold, kosztami transakcyjnymi, stabilnością walk-forward i ryzykiem przeoptymalizowania.

