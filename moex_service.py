"""FastAPI service for MOEX candle analysis."""

import datetime as dt
import io
from typing import List, Optional

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

MOEX_URL_TEMPLATE = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/"
    "securities/{ticker}/candles.csv"
)

app = FastAPI(
    title="MOEX Candle Analysis Service",
    description=(
        "Сервис позволяет рассчитывать изменение цены между выбранными "
        "временными отметками для заданного тикера и диапазона дат."
    ),
)


class CandleDiff(BaseModel):
    date: dt.date = Field(..., description="Дата торгового дня")
    start_price: float = Field(..., description="Цена в момент start_time")
    end_price: float = Field(..., description="Цена в момент end_time")
    diff_abs: float = Field(..., description="Абсолютная разница цен")
    diff_pct: Optional[float] = Field(
        None, description="Процентная разница относительно start_price"
    )


class DiffResponse(BaseModel):
    ticker: str
    board: str
    interval: int = Field(..., description="Интервал свечей в минутах")
    start_time: dt.time
    end_time: dt.time
    results: List[CandleDiff]


def _parse_time(value: str, name: str) -> dt.time:
    try:
        return dt.datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {name} format. Use HH:MM") from exc


def fetch_candles(
    *,
    ticker: str,
    board: str,
    interval: int,
    date_from: dt.date,
    date_till: dt.date,
) -> pd.DataFrame:
    url = MOEX_URL_TEMPLATE.format(board=board, ticker=ticker)
    base_params = {
        "from": date_from.isoformat(),
        "till": date_till.isoformat(),
        "interval": str(interval),
    }

    frames: List[pd.DataFrame] = []
    start = 0

    while True:
        params = {**base_params, "start": start}
        response = requests.get(url, params=params)
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Ticker or board not found on MOEX")
        response.raise_for_status()

        chunk = pd.read_csv(io.StringIO(response.text), sep=";", skiprows=2)
        if chunk.empty:
            break

        frames.append(chunk)
        start += len(chunk)

    if not frames:
        raise HTTPException(status_code=404, detail="No candles found for given parameters")

    df = pd.concat(frames, ignore_index=True)
    df["begin"] = pd.to_datetime(df["begin"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["begin", "close"])
    df["date"] = df["begin"].dt.date
    return df


@app.get("/diff", response_model=DiffResponse)
def get_candle_diff(
    ticker: str = Query("TGLD", description="Код бумаги на MOEX"),
    board: str = Query("TQTF", description="Идентификатор режима торгов"),
    date_from: dt.date = Query(..., description="Начальная дата (YYYY-MM-DD)"),
    date_till: dt.date = Query(..., description="Конечная дата (YYYY-MM-DD)"),
    start_time: str = Query("09:55", description="Стартовое время HH:MM"),
    end_time: str = Query("10:15", description="Конечное время HH:MM"),
    interval: int = Query(1, ge=1, le=60, description="Интервал свечей в минутах"),
):
    if date_from > date_till:
        raise HTTPException(status_code=400, detail="date_from must be earlier than date_till")

    start_time_obj = _parse_time(start_time, "start_time")
    end_time_obj = _parse_time(end_time, "end_time")

    if start_time_obj >= end_time_obj:
        raise HTTPException(status_code=400, detail="start_time must be earlier than end_time")

    df = fetch_candles(
        ticker=ticker,
        board=board,
        interval=interval,
        date_from=date_from,
        date_till=date_till,
    )

    results: List[CandleDiff] = []

    for day, chunk in df.groupby("date"):
        base_dt = dt.datetime.combine(day, dt.time())
        ts_start = pd.Timestamp(
            base_dt.replace(hour=start_time_obj.hour, minute=start_time_obj.minute)
        )
        ts_end = pd.Timestamp(
            base_dt.replace(hour=end_time_obj.hour, minute=end_time_obj.minute)
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

    return DiffResponse(
        ticker=ticker,
        board=board,
        interval=interval,
        start_time=start_time_obj,
        end_time=end_time_obj,
        results=results,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("moex_service:app", host="0.0.0.0", port=8000, reload=False)
