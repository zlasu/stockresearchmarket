from __future__ import annotations

from typer.testing import CliRunner

from stockresearchmarket.cli import app


def main() -> None:
    result = CliRunner().invoke(app, ["smoke"])
    print(result.output)
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()

