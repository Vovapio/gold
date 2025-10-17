"""Command-line utility for MOEX candle analysis.

The script downloads intraday candles for a given ticker/board from MOEX,
calculates the price difference between two time marks for every trading day
in the selected period and stores the result as a CSV table. A formatted view
of the table is also printed to the terminal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import requests

MOEX_URL_TEMPLATE = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/"
    "securities/{ticker}/candles.csv"
)


@dataclass
class CandleDiff:
    """Stores price information for a single trading day."""

    date: dt.date
    start_price: float
    end_price: float
    diff_abs: float
    diff_pct: Optional[float]


def parse_time(value: str) -> dt.time:
    """Parse HH:MM time strings for CLI arguments."""

    try:
        return dt.datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:  # pragma: no cover - handled as CLI error
        raise argparse.ArgumentTypeError(
            f"Некорректный формат времени '{value}'. Используйте HH:MM"
        ) from exc


def parse_date(value: str) -> dt.date:
    """Parse YYYY-MM-DD date strings for CLI arguments."""

    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - handled as CLI error
        raise argparse.ArgumentTypeError(
            f"Некорректный формат даты '{value}'. Используйте YYYY-MM-DD"
        ) from exc


def prompt_date(message: str) -> dt.date:
    """Interactively request a date value from the user."""

    while True:
        try:
            raw = input(message).strip()
        except EOFError as exc:  # pragma: no cover - defensive fallback
            raise SystemExit(1) from exc

        if not raw:
            print("Поле не может быть пустым. Повторите ввод.")
            continue

        try:
            return parse_date(raw)
        except argparse.ArgumentTypeError as exc:
            print(exc)


def fetch_candles(
    *,
    ticker: str,
    board: str,
    interval: int,
    date_from: dt.date,
    date_till: dt.date,
    chunk_days: int = 5,
) -> pd.DataFrame:
    """Load all intraday candles for the requested period."""

    if chunk_days < 1:
        raise ValueError("chunk_days должен быть положительным")

    url = MOEX_URL_TEMPLATE.format(board=board, ticker=ticker)
    frames: List[pd.DataFrame] = []

    for chunk_start, chunk_end in _iterate_date_ranges(date_from, date_till, chunk_days):
        params = {
            "from": chunk_start.isoformat(),
            "till": chunk_end.isoformat(),
            "interval": str(interval),
        }

        try:
            frames.extend(_fetch_paginated(url, params))
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ошибка при обращении к MOEX: "
                f"{exc} (диапазон {chunk_start}..{chunk_end})"
            ) from exc

    if not frames:
        raise ValueError("Не удалось получить свечи по заданным параметрам")

    df = pd.concat(frames, ignore_index=True)
    df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["begin", "close"])
    df["date"] = df["begin"].dt.date
    return df


def _iterate_date_ranges(
    date_from: dt.date, date_till: dt.date, chunk_days: int
) -> Iterator[Tuple[dt.date, dt.date]]:
    """Yield inclusive date ranges split into chunks of `chunk_days`."""

    current = date_from
    delta = dt.timedelta(days=chunk_days - 1)

    while current <= date_till:
        chunk_end = min(current + delta, date_till)
        yield current, chunk_end
        current = chunk_end + dt.timedelta(days=1)


def _fetch_paginated(url: str, params: dict) -> List[pd.DataFrame]:
    """Fetch a paginated MOEX candle response for a single date range."""

    frames: List[pd.DataFrame] = []
    start = 0

    while True:
        page_params = {**params, "start": start}
        response = requests.get(url, params=page_params, timeout=15)

        if response.status_code == 404:
            raise ValueError("Тикер или режим торгов не найдены на MOEX")

        response.raise_for_status()

        chunk = pd.read_csv(io.StringIO(response.text), sep=";", skiprows=2)
        if chunk.empty:
            break

        frames.append(chunk)
        start += len(chunk)

    return frames


def calculate_diffs(
    df: pd.DataFrame, *, start_time: dt.time, end_time: dt.time
) -> List[CandleDiff]:
    """Calculate price differences for each trading day."""

    results: List[CandleDiff] = []

    for day, chunk in df.groupby("date"):
        base_dt = dt.datetime.combine(day, dt.time())
        ts_start = pd.Timestamp(
            base_dt.replace(hour=start_time.hour, minute=start_time.minute)
        )
        ts_end = pd.Timestamp(
            base_dt.replace(hour=end_time.hour, minute=end_time.minute)
        )

        price_start = chunk.loc[chunk["begin"] == ts_start, "close"]
        if price_start.empty:
            price_start = chunk.loc[chunk["begin"] >= ts_start, "close"].head(1)

        price_end = chunk.loc[chunk["begin"] == ts_end, "close"]
        if price_end.empty:
            price_end = chunk.loc[chunk["begin"] <= ts_end, "close"].tail(1)

        if price_start.empty or price_end.empty:
            continue

        start_val = float(price_start.iloc[0])
        end_val = float(price_end.iloc[0])
        diff_abs = end_val - start_val
        diff_pct = (diff_abs / start_val) * 100 if start_val else None

        results.append(
            CandleDiff(
                date=day,
                start_price=start_val,
                end_price=end_val,
                diff_abs=diff_abs,
                diff_pct=diff_pct,
            )
        )

    return sorted(results, key=lambda item: item.date)


def results_to_dataframe(results: Iterable[CandleDiff]) -> pd.DataFrame:
    """Convert a sequence of results to a pandas DataFrame."""

    df = pd.DataFrame(
        [
            {
                "date": item.date,
                "start_price": item.start_price,
                "end_price": item.end_price,
                "diff_abs": item.diff_abs,
                "diff_pct": item.diff_pct,
            }
            for item in results
        ]
    )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    return df


def build_output_path(
    *,
    provided: Optional[str],
    ticker: str,
    board: str,
    date_from: dt.date,
    date_till: dt.date,
    start_time: dt.time,
    end_time: dt.time,
) -> Path:
    """Derive the output CSV filename."""

    if provided:
        return Path(provided)

    slug = (
        f"{ticker}_{board}_{date_from.isoformat()}_{date_till.isoformat()}_"
        f"{start_time.strftime('%H%M')}_{end_time.strftime('%H%M')}"
    )
    return Path(f"moex_diffs_{slug}.csv")


def print_table(df: pd.DataFrame) -> None:
    """Pretty-print a dataframe with numeric formatting."""

    if df.empty:
        print("⚠️  Нет данных для отображения по выбранным параметрам.")
        return

    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.date
    display_df["diff_pct"] = display_df["diff_pct"].map(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
    )

    def _fmt(value: float) -> str:
        return f"{value:,.2f}".replace(",", " ")

    formatters = {
        "start_price": _fmt,
        "end_price": _fmt,
        "diff_abs": _fmt,
    }

    print("\nРезультаты:")
    print(display_df.to_string(index=False, formatters=formatters))


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Persist the dataframe as CSV with readable floats."""

    path.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()
    if df_to_save.empty:
        df_to_save = pd.DataFrame(
            columns=["date", "start_price", "end_price", "diff_abs", "diff_pct"]
        )
    else:
        df_to_save["date"] = df_to_save["date"].dt.date

    df_to_save.to_csv(path, index=False, float_format="%.6f")


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Загружает минутные свечи MOEX и считает изменение цены между двумя "
            "временными отметками для каждого торгового дня."
        )
    )

    parser.add_argument("--ticker", default="TGLD", help="Код бумаги на MOEX (например, SBER)")
    parser.add_argument(
        "--board",
        default="TQTF",
        help="Режим торгов MOEX (например, TQBR для основных акций)",
    )
    parser.add_argument(
        "--date-from",
        type=parse_date,
        help="Начальная дата периода в формате YYYY-MM-DD",
    )
    parser.add_argument(
        "--date-till",
        type=parse_date,
        help="Конечная дата периода в формате YYYY-MM-DD",
    )
    parser.add_argument(
        "--start-time",
        type=parse_time,
        default=parse_time("09:55"),
        help="Начальное время (HH:MM)",
    )
    parser.add_argument(
        "--end-time",
        type=parse_time,
        default=parse_time("10:15"),
        help="Конечное время (HH:MM)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        choices=range(1, 61),
        metavar="[1-60]",
        help="Интервал свечей в минутах",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Путь к CSV-файлу с результатами (по умолчанию формируется автоматически)",
    )

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entry point used by the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.date_from is None:
        args.date_from = prompt_date("Введите начальную дату (YYYY-MM-DD): ")

    if args.date_till is None:
        args.date_till = prompt_date("Введите конечную дату (YYYY-MM-DD): ")

    if args.date_from > args.date_till:
        parser.error("--date-from не может быть больше, чем --date-till")

    if args.start_time >= args.end_time:
        parser.error("--start-time должен быть раньше, чем --end-time")

    print(
        "⏳ Загружаем данные:",
        f"тикер={args.ticker}",
        f"режим={args.board}",
        f"даты={args.date_from}..{args.date_till}",
        f"интервал={args.interval} мин",
    )

    try:
        candles = fetch_candles(
            ticker=args.ticker,
            board=args.board,
            interval=args.interval,
            date_from=args.date_from,
            date_till=args.date_till,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    diffs = calculate_diffs(
        candles, start_time=args.start_time, end_time=args.end_time
    )
    table = results_to_dataframe(diffs)

    output_path = build_output_path(
        provided=args.output,
        ticker=args.ticker,
        board=args.board,
        date_from=args.date_from,
        date_till=args.date_till,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    save_table(table, output_path)
    print(f"💾 Таблица сохранена в {output_path.resolve()}")

    print_table(table)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI execution path
    sys.exit(main())
