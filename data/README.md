# Data Workspace

- `historical/` - cache Parquet znormalizowanych danych OHLCV.
- `raw/` - ręcznie dodane CSV, np. `data/raw/SPY.csv`.
- `external/` - eksporty z płatnych lub alternatywnych źródeł danych.

Dane historyczne i surowe CSV są ignorowane przez Git, żeby nie mieszać kodu z cache.

