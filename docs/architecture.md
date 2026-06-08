# Architektura

`StockResearchMarket` ma być laboratorium badawcze, a nie bot transakcyjny. Dane, strategie, silnik, optymalizacja i raportowanie są rozdzielone tak, żeby łatwo dodać nowe źródło danych albo rodzinę strategii.

```text
configs/                 Konfiguracje badań i siatki parametrów
data/historical/          Lokalny cache Parquet
experiments/              Wyniki konkretnych uruchomień
stockresearchmarket/
  data/                   Źródła danych i normalizacja OHLCV
  features/               Wskaźniki techniczne
  strategies/             Sygnały i registry strategii
  engine/                 Backtest jednoaktywny i portfelowy
  optimization/           Grid, Optuna, walk-forward
  reports/                Raporty HTML/CSV
```

## Konwencja Backtestu

- Sygnał z dnia `t` jest wykonywany od kolejnej świecy, żeby uniknąć look-ahead bias.
- Ceny domyślnie używają skorygowanego close, gdy źródło je udostępnia.
- Koszt transakcyjny to `fee_bps + slippage_bps + spread_bps / 2` od zmiany ekspozycji.
- Każdy wynik jest porównywany z buy and hold na tym samym instrumencie i zakresie.
- Optymalizację należy czytać razem z walk-forward, liczbą transakcji i stabilnością parametrów.

