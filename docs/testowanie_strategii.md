# Testowanie Strategii

Minimalny proces dla sensownego wyniku:

1. Uruchom szybki backtest z domyślnymi parametrami.
2. Porównaj z buy and hold na tym samym tickerze i zakresie dat.
3. Uruchom grid/Optuna, ale traktuj najlepszy wynik jako hipotezę, nie dowód.
4. Sprawdź walk-forward i stabilność parametrów.
5. Odrzuć strategie z małą liczbą transakcji, skrajną zależnością od jednego okresu albo dużą wrażliwością na koszty.

Przykłady:

```bash
uv run stockresearch run --strategy rsi_mean_reversion --tickers SPY,QQQ --years 20
uv run stockresearch optimize --strategy donchian_breakout --ticker QQQ --method optuna --years 20
uv run stockresearch portfolio --universe core,sectors --years 20
```

