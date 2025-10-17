import io
from datetime import datetime, timedelta

import pandas as pd
import requests

# === Настройки ===
TICKER = "TGLD"
BOARD = "TQTF"
DAYS_BACK = 30  # сколько дней брать (примерно месяц)

# === Формируем даты ===
till = datetime.now().date()
from_ = till - timedelta(days=DAYS_BACK)

url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/{BOARD}/securities/{TICKER}/candles.csv"
base_params = {"from": from_.isoformat(), "till": till.isoformat(), "interval": "1"}

print(f"⏳ Загружаем данные за период {from_} — {till}...")


def load_all_pages() -> pd.DataFrame:
    """Скачиваем все страницы свечек (по умолчанию MOEX отдаёт ~500 строк за раз)."""

    frames = []
    start = 0

    while True:
        params = {**base_params, "start": start}
        r = requests.get(url, params=params)
        r.raise_for_status()

        chunk = pd.read_csv(io.StringIO(r.text), sep=";", skiprows=2)
        if chunk.empty:
            break

        frames.append(chunk)
        start += len(chunk)
        print(f"  → Получено {len(chunk)} строк (start={start})")

    if not frames:
        raise ValueError("Не удалось получить данные от MOEX")

    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv("candles_tgld.csv", sep=";", index=False)
    return df_all


df = load_all_pages()

# === Читаем и чистим ===
df["begin"] = pd.to_datetime(df["begin"])
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df = df.dropna(subset=["begin", "close"])
df["date"] = df["begin"].dt.date

# === Расчёт разницы 10:00 → 10:10 ===
results = []
for d, chunk in df.groupby("date"):
    t_start = pd.Timestamp(datetime.combine(d, datetime.min.time()).replace(hour=10, minute=0))
    t_end = t_start.replace(minute=10)

    p10 = chunk.loc[chunk["begin"] == t_start, "close"]
    if p10.empty:
        p10 = chunk.loc[chunk["begin"] >= t_start, "close"].head(1)
    p1010 = chunk.loc[chunk["begin"] == t_end, "close"]
    if p1010.empty:
        p1010 = chunk.loc[chunk["begin"] <= t_end, "close"].tail(1)

    if p10.empty or p1010.empty:
        continue

    p10v, p1010v = float(p10.iloc[0]), float(p1010.iloc[0])
    diff = p1010v - p10v
    diff_pct = (diff / p10v) * 100 if p10v != 0 else None

    results.append({
        "date": d,
        "price_10_00": p10v,
        "price_10_10": p1010v,
        "diff_abs": diff,
        "diff_pct": diff_pct
    })

out = pd.DataFrame(results).sort_values("date")
out.to_csv("tgld_10_00_to_10_10_diffs.csv", index=False)

print("\n✅ Готово! Результаты сохранены в tgld_10_00_to_10_10_diffs.csv")
print(out.head(10))