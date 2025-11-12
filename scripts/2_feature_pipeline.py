#!/usr/bin/env python3
import sys
import os
import json
import warnings
from pathlib import Path
from datetime import date
import re

warnings.filterwarnings("ignore", module="IPython")

# ---------- Paths / PYTHONPATH ----------
root_dir = Path().absolute()
if root_dir.parts[-1:] == ("airquality",):
    root_dir = Path(*root_dir.parts[:-1])
if root_dir.parts[-1:] == ("notebooks",):
    root_dir = Path(*root_dir.parts[:-1])
root_dir = root_dir.resolve()
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
print(f"Local environment — project root: {root_dir}")

if root_dir not in sys.path:
    sys.path.append(root_dir)
    print(f"Added to PYTHONPATH: {root_dir}")

# ---------- Settings ----------
from mlfs import config
settings = config.HopsworksSettings(_env_file=str(root_dir / ".env"))

# ---------- Imports ----------
import pandas as pd
import numpy as np
import hopsworks
from mlfs.airquality import util

# ---------- Hopsworks login ----------
project = hopsworks.login(engine="python")
fs = project.get_feature_store()
secrets = hopsworks.get_secrets_api()

# ---------- Helpers ----------
def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text

def get_aqicn_api_key() -> str:
    if settings.AQICN_API_KEY is not None:
        return settings.AQICN_API_KEY.get_secret_value()
    try:
        return secrets.get_secret("AQICN_API_KEY").value
    except Exception:
        raise RuntimeError("AQICN_API_KEY not found in .env or Hopsworks secrets")

def get_sensor_rows(sensors_csv: str) -> pd.DataFrame:
    req = {"AQICN_URL", "country", "city", "street", "latitude", "longitude"}
    df = pd.read_csv(sensors_csv, dtype=str).fillna("")
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in sensors CSV: {sorted(missing)}")
    return df

def process_sensor(row: pd.Series, aq_api_key: str, today: date) -> None:
    aqicn_url = str(row["AQICN_URL"]).strip()
    country   = str(row["country"]).strip()
    city      = str(row["city"]).strip()
    street    = str(row["street"]).strip()
    latitude  = str(row["latitude"]).strip()
    longitude = str(row["longitude"]).strip()
    street_slug = slugify(street)

    print(f"\n=== Daily update: {city} / {street} ({today}) ===")

    # Retrieve today's air quality
    print("get pm25 called")
    aq_today_df = util.get_pm25(aqicn_url, country, city, street, today, aq_api_key)
    aq_today_df["pm25"] = aq_today_df["pm25"].astype(np.double)
    aq_today_df["country"] = country
    aq_today_df["city"] = city
    aq_today_df["street"] = street
    aq_today_df["url"] = aqicn_url

    # Retrieve daily weather (sample around noon)
    print("get hourly weather called")
    hourly_df = util.get_hourly_weather_forecast(city, latitude, longitude).set_index("date")
    daily_df = hourly_df.between_time("11:59", "12:01").reset_index()
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["city"] = city

    # Per-sensor feature groups
    aq_fg = fs.get_feature_group(name=f"air_quality_{street_slug}", version=1)
    wx_fg = fs.get_feature_group(name=f"weather_{street_slug}", version=1)

    # Inserts
    aq_fg.insert(aq_today_df, wait=True)
    wx_fg.insert(daily_df, wait=True)

    print(f"✓ Completed: {city} / {street}")

# ---------- Main ----------
def main():
    aq_api_key = get_aqicn_api_key()
    sensors_csv = os.environ.get("SENSORS_CSV", str(root_dir / "data" / "sensors.csv"))
    if not Path(sensors_csv).exists():
        raise FileNotFoundError(f"Missing sensors CSV: {sensors_csv}")

    sensors_df = get_sensor_rows(sensors_csv)
    today = date.today()
    print(f"Processing {len(sensors_df)} sensors from {sensors_csv}")

    for _, row in sensors_df.iterrows():
        try:
            process_sensor(row, aq_api_key, today)
        except Exception as e:
            city = row.get("city", "")
            street = row.get("street", "")
            print(f"! Error processing {city} / {street}: {e}")

    print("\nAll sensors processed.")

if __name__ == "__main__":
    main()
