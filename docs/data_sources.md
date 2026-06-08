# Źródła Danych

## Domyślne

- `yfinance`: szybkie dane dzienne OHLCV, dobre do prototypowania strategii.
- `stooq`: fallback dla wielu tickerów akcyjnych i ETF-ów.
- `csv`: własne pliki w `data/raw/<TICKER>.csv`, z kolumnami `date/open/high/low/close/volume`.
- `synthetic`: deterministyczny tryb offline dla testów i CI.

## Premium Do Dodania

Projekt ma przygotowane miejsce na integracje wymagające kluczy API:

- Polygon.io dla split-adjusted bars, fundamentals i danych intraday;
- Nasdaq Data Link dla danych makro i alternatywnych;
- Alpha Vantage jako prosty backup;
- Interactive Brokers / Alpaca jako późniejsze źródła paper/live, nie do researchu historycznego.

## Jakość Danych

Przed interpretacją strategii sprawdzaj:

- faktyczny zakres dat po pobraniu;
- braki w OHLCV;
- survivorship bias przy koszykach akcji;
- corporate actions i dywidendy;
- czy benchmark ma dokładnie ten sam zakres dat.

