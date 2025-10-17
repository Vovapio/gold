import pandas as pd
import requests
from datetime import datetime, timedelta

# === Настройки ===
TICKER = "TGLD"
BOARD = "TQTF"
DAYS_BACK = 120  # берём несколько месяцев, чтобы видеть месячные изменения

# === Формируем даты ===
till = datetime.now().date()
from_ = till - timedelta(days=DAYS_BACK)

url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/{BOARD}/securities/{TICKER}/candles.csv"
params = {"from": from_.isoformat(), "till": till.isoformat(), "interval": "1"}

print(f"⏳ Загружаем данные за период {from_} — {till}...")

r = requests.get(url, params=params)
r.raise_for_status()

with open("candles_tgld.csv", "wb") as f:
    f.write(r.content)

# === Читаем и чистим ===
df = pd.read_csv("candles_tgld.csv", sep=";", skiprows=1)
if "begin" not in df.columns or "close" not in df.columns:
    raise ValueError(f"Не найдены нужные столбцы: {df.columns.tolist()}")

df["begin"] = pd.to_datetime(df["begin"])
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df = df.dropna(subset=["begin", "close"]).sort_values("begin")

df["month"] = df["begin"].dt.to_period("M")

# === Расчёт месячных скачков цены ===
results = []
for period, chunk in df.groupby("month"):
    chunk = chunk.sort_values("begin")
    first_close = float(chunk.iloc[0]["close"])
    last_close = float(chunk.iloc[-1]["close"])
    diff = last_close - first_close
    diff_pct = (diff / first_close) * 100 if first_close != 0 else None

    results.append({
        "month": period.to_timestamp().date(),
        "price_start": first_close,
        "price_end": last_close,
        "diff_abs": diff,
        "diff_pct": diff_pct,
    })

out = pd.DataFrame(results).sort_values("month")
out.to_csv("tgld_monthly_price_jumps.csv", index=False)

print("\n✅ Готово! Результаты сохранены в tgld_monthly_price_jumps.csv")
print(out)
