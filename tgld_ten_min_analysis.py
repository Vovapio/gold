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

# === Параметры расчёта интервала ===
START_HOUR, START_MINUTE = 9, 55
END_HOUR, END_MINUTE = 10, 15
OUTPUT_FILE = "tgld_09_55_to_10_15_diffs.csv"

# === Расчёт разницы 9:55 → 10:15 ===
results = []
for d, chunk in df.groupby("date"):
    base_time = datetime.combine(d, datetime.min.time())
    t_start = pd.Timestamp(base_time.replace(hour=START_HOUR, minute=START_MINUTE))
    t_end = pd.Timestamp(base_time.replace(hour=END_HOUR, minute=END_MINUTE))

    p_start = chunk.loc[chunk["begin"] == t_start, "close"]
    if p_start.empty:
        p_start = chunk.loc[chunk["begin"] >= t_start, "close"].head(1)
    p_end = chunk.loc[chunk["begin"] == t_end, "close"]
    if p_end.empty:
        p_end = chunk.loc[chunk["begin"] <= t_end, "close"].tail(1)

    if p_start.empty or p_end.empty:
        continue

    p_start_v, p_end_v = float(p_start.iloc[0]), float(p_end.iloc[0])
    diff = p_end_v - p_start_v
    diff_pct = (diff / p_start_v) * 100 if p_start_v != 0 else None

    results.append({
        "date": d,
        "price_09_55": p_start_v,
        "price_10_15": p_end_v,
        "diff_abs": diff,
        "diff_pct": diff_pct
    })

out = pd.DataFrame(results).sort_values("date")
out.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Готово! Результаты сохранены в {OUTPUT_FILE}")
print(out.head(10))
