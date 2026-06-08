.PHONY: setup test lint smoke data optimize portfolio

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

