.PHONY: setup test lint smoke data optimize portfolio garp garp-auto sp500-current

setup:
	uv sync --extra dev --python 3.11.11

test:
	uv run pytest

lint:
	uv run ruff check .

smoke:
	uv run stockresearch smoke

data:
	uv run stockresearch data --years 20

optimize:
	uv run stockresearch optimize --strategy sma_cross --ticker SPY --method grid --years 20

portfolio:
	uv run stockresearch portfolio --years 20

garp:
	uv run stockresearch garp-run --experiment 001_baseline_garp --provider synthetic --years 10

garp-auto:
	uv run stockresearch garp-autoresearch --provider synthetic --years 8 --max-experiments 6

sp500-current:
	uv run python scripts/run_current_sp500_price_tests.py
