"""Interactive terminal tool for MOEX candle analysis.

The script asks the user for a date and time range along with the fund (ticker),
downloads intraday candles from MOEX, calculates the price difference between
two time marks for every trading day and stores the result as a CSV table. A
formatted view of the table is also printed to the terminal.
"""

from __future__ import annotations

import datetime as dt
import io
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import requests
from requests import Timeout

MOEX_URL_TEMPLATE = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/"
    "securities/{ticker}/candles.csv"
)

DEFAULT_BOARD = "TQTF"
DEFAULT_INTERVAL = 1


@dataclass
class CandleDiff:
    """Stores price information for a single trading day."""

    date: dt.date
    start_price: float
    end_price: float
    diff_abs: float
    diff_pct: Optional[float]


def parse_time(value: str) -> dt.time:
    """Parse HH:MM time strings."""

    try:
        return dt.datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:  # pragma: no cover - handled via prompt loop
        raise ValueError(f"Некорректный формат времени '{value}'. Используйте HH:MM") from exc


def parse_date(value: str) -> dt.date:
    """Parse YYYY-MM-DD date strings."""

    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - handled via prompt loop
        raise ValueError(
            f"Некорректный формат даты '{value}'. Используйте YYYY-MM-DD"
        ) from exc


def prompt_date(message: str) -> dt.date:
    """Interactively request a date value from the user."""

    while True:
        try:
            raw = input(message).strip()
        except EOFError:
            print("\nВвод прерван. Выход.")
            raise SystemExit(1) from None

        if not raw:
            print("Поле не может быть пустым. Повторите ввод.")
            continue

        try:
            return parse_date(raw)
        except ValueError as exc:
            print(exc)


def prompt_time(message: str) -> dt.time:
    """Interactively request a time value from the user."""

    while True:
        try:
            raw = input(message).strip()
        except EOFError:
            print("\nВвод прерван. Выход.")
            raise SystemExit(1) from None

        if not raw:
            print("Поле не может быть пустым. Повторите ввод.")
            continue

        try:
            return parse_time(raw)
        except ValueError as exc:
            print(exc)


def prompt_ticker(message: str, *, default: Optional[str] = None) -> str:
    """Request a ticker (fund code) from the user."""

    while True:
        try:
            raw = input(message).strip().upper()
        except EOFError:
            print("\nВвод прерван. Выход.")
            raise SystemExit(1) from None

        if not raw and default:
            return default

        if not raw:
            print("Название фонда не может быть пустым. Повторите ввод.")
            continue

        return raw


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

    ranges = deque(_iterate_date_ranges(date_from, date_till, chunk_days))

    while ranges:
        chunk_start, chunk_end = ranges.popleft()
        params = {
            "from": chunk_start.isoformat(),
            "till": chunk_end.isoformat(),
            "interval": str(interval),
        }

        try:
            frames.extend(_fetch_paginated(url, params))
        except Timeout as exc:
            days = (chunk_end - chunk_start).days + 1

            if days == 1:
                raise RuntimeError(
                    "Ошибка при обращении к MOEX: "
                    f"{exc} (диапазон {chunk_start}..{chunk_end})"
                ) from exc

            offset = max(days // 2, 1) - 1
            split_point = chunk_start + dt.timedelta(days=offset)

            first_range = (chunk_start, min(split_point, chunk_end))
            second_start = split_point + dt.timedelta(days=1)

            print(
                "⏱️  Таймаут при запросе диапазона",
                f"{chunk_start}..{chunk_end}.",
                "Пробуем меньшие интервалы...",
            )

            if second_start <= chunk_end:
                ranges.appendleft((second_start, chunk_end))
            if first_range[0] <= first_range[1]:
                ranges.appendleft(first_range)
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
    ticker: str,
    board: str,
    date_from: dt.date,
    date_till: dt.date,
    start_time: dt.time,
    end_time: dt.time,
) -> Path:
    """Derive the output CSV filename."""

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


def prompt_date_range() -> Tuple[dt.date, dt.date]:
    """Request a valid date interval from the user."""

    while True:
        start = prompt_date("Введите начальную дату (YYYY-MM-DD): ")
        end = prompt_date("Введите конечную дату (YYYY-MM-DD): ")

        if start > end:
            print("Начальная дата не может быть позже конечной. Повторите ввод.")
            continue

        return start, end


def prompt_time_range() -> Tuple[dt.time, dt.time]:
    """Request a valid time interval from the user."""

    while True:
        start = prompt_time("Введите время начала (HH:MM): ")
        end = prompt_time("Введите время конца (HH:MM): ")

        if start >= end:
            print("Время начала должно быть раньше времени конца. Повторите ввод.")
            continue

        return start, end


def main() -> int:
    """Entry point used by the interactive CLI."""

    print("Введите параметры для анализа свечей MOEX:")

    date_from, date_till = prompt_date_range()
    start_time, end_time = prompt_time_range()
    ticker = prompt_ticker("Введите название фонда (тикер): ")
    board = prompt_ticker(
        (
            "Введите режим торгов (например, TQTF, Enter для значения по умолчанию"
            f" {DEFAULT_BOARD}): "
        ),
        default=DEFAULT_BOARD,
    )

    interval = DEFAULT_INTERVAL

    print(
        "⏳ Загружаем данные:",
        f"тикер={ticker}",
        f"режим={board}",
        f"даты={date_from}..{date_till}",
        f"интервал={interval} мин",
    )

    try:
        candles = fetch_candles(
            ticker=ticker,
            board=board,
            interval=interval,
            date_from=date_from,
            date_till=date_till,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    diffs = calculate_diffs(candles, start_time=start_time, end_time=end_time)
    table = results_to_dataframe(diffs)

    output_path = build_output_path(
        ticker=ticker,
        board=board,
        date_from=date_from,
        date_till=date_till,
        start_time=start_time,
        end_time=end_time,
    )
    save_table(table, output_path)
    print(f"💾 Таблица сохранена в {output_path.resolve()}")

    print_table(table)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI execution path
    sys.exit(main())
