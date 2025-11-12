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
import great_expectations as ge
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

def air_quality_csv_path(street: str) -> Path:
    """Convention: data/<street_slug>.csv"""
    return root_dir / "data" / f"{slugify(street)}.csv"

def save_secret(name: str, value: str) -> None:
    try:
        existing = secrets.get_secret(name)
        if existing is not None:
            existing.delete()
            print(f"Replacing existing secret: {name}")
    except Exception:
        pass
    secrets.create_secret(name, value)

def ge_suite_air_quality() -> ge.core.ExpectationSuite:
    suite = ge.core.ExpectationSuite("aq_expectation_suite")
    suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_min_to_be_between",
            kwargs={"column": "pm25", "min_value": -0.1, "max_value": 500.0, "strict_min": True},
        )
    )
    return suite

def ge_suite_weather() -> ge.core.ExpectationSuite:
    suite = ge.core.ExpectationSuite("weather_expectation_suite")
    def add_min_gt_zero(col: str):
        suite.add_expectation(
            ge.core.ExpectationConfiguration(
                expectation_type="expect_column_min_to_be_between",
                kwargs={"column": col, "min_value": -0.1, "max_value": 1000.0, "strict_min": True},
            )
        )
    add_min_gt_zero("precipitation_sum")
    add_min_gt_zero("wind_speed_10m_max")
    return suite

def process_sensor(row: pd.Series, aq_api_key: str, today: date) -> None:
    aqicn_url = str(row["AQICN_URL"]).strip()
    country   = str(row["country"]).strip()
    city      = str(row["city"]).strip()
    street    = str(row["street"]).strip()
    latitude  = str(row["latitude"]).strip()
    longitude = str(row["longitude"]).strip()

    street_slug = slugify(street)
    print(f"\n=== Processing sensor: {city} / {street} ===")


    # Load historical AQ CSV
    aq_csv = air_quality_csv_path(street)
    util.check_file_path(str(aq_csv))
    df = pd.read_csv(aq_csv, parse_dates=["date"], skipinitialspace=True)

    df_aq = df[["date", "pm25"]].copy()
    df_aq["pm25"] = df_aq["pm25"].astype("float32")
    df_aq.dropna(inplace=True)
    df_aq["country"] = country
    df_aq["city"] = city
    df_aq["street"] = street
    df_aq["url"] = aqicn_url

    # Historical weather (from earliest AQ date to today)
    earliest_aq_date = pd.Series.min(df_aq["date"]).strftime("%Y-%m-%d")
    weather_df = util.get_historical_weather(city, earliest_aq_date, str(today), latitude, longitude)

    # GE expectation suites
    aq_suite = ge_suite_air_quality()
    weather_suite = ge_suite_weather()

    # Per-sensor secrets
    # Global API key (one time overall is fine, but harmless to overwrite)
    save_secret("AQICN_API_KEY", aq_api_key)

    # Per-sensor location JSON
    location_json = json.dumps({
        "country": country,
        "city": city,
        "street": street,
        "aqicn_url": aqicn_url,
        "latitude": latitude,
        "longitude": longitude,
    })
    save_secret(f"SENSOR_LOCATION_JSON_{street_slug}", location_json)

    # Feature Groups (per sensor, suffixed)
    aq_fg_name = f"air_quality_{street_slug}"
    weather_fg_name = f"weather_{street_slug}"

    air_quality_fg = fs.get_or_create_feature_group(
        name=aq_fg_name,
        description=f"Air Quality per day ({street}, {city})",
        version=1,
        primary_key=["country", "city", "street"],
        event_time="date",
        expectation_suite=aq_suite,
    )

    df_aq["pm25"] = df_aq["pm25"].astype(np.double)
    air_quality_fg.insert(df_aq)
    #air_quality_fg.update_feature_description("date", "Date of measurement of air quality")
    #air_quality_fg.update_feature_description("country", "Country (sometimes a city in aqicn.org)")
    #air_quality_fg.update_feature_description("city", "City of measurement")
    #air_quality_fg.update_feature_description("street", "Street of measurement")
    #air_quality_fg.update_feature_description("pm25", "PM2.5 (μg/m3)")

    weather_fg = fs.get_or_create_feature_group(
        name=weather_fg_name,
        description=f"Weather per day ({street}, {city})",
        version=1,
        primary_key=["city"],
        event_time="date",
        expectation_suite=weather_suite,
    )
    weather_fg.insert(weather_df, wait=True)
    #weather_fg.update_feature_description("date", "Date of weather measurement")
    #weather_fg.update_feature_description("city", "City for weather")
    #weather_fg.update_feature_description("temperature_2m_mean", "Temperature in Celsius")
    #weather_fg.update_feature_description("precipitation_sum", "Precipitation (mm)")
    #weather_fg.update_feature_description("wind_speed_10m_max", "Max wind speed at 10m")
    #weather_fg.update_feature_description("wind_direction_10m_dominant", "Dominant wind direction")

    print(f"✓ Completed: {city} / {street}")

# ---------- Main ----------
def main():
    if settings.AQICN_API_KEY is None:
        print("You need to set AQICN_API_KEY in .env")
        sys.exit(1)

    aq_api_key = settings.AQICN_API_KEY.get_secret_value()
    sensors_csv = os.environ.get("SENSORS_CSV", str(root_dir / "data" / "sensors.csv"))

    # Expected columns: AQICN_URL,country,city,street,latitude,longitude
    if not Path(sensors_csv).exists():
        print(f"Missing sensors CSV: {sensors_csv}")
        sys.exit(1)

    sensors_df = pd.read_csv(sensors_csv, dtype=str).fillna("")
    required_cols = {"AQICN_URL", "country", "city", "street", "latitude", "longitude"}
    missing = required_cols - set(sensors_df.columns)
    if missing:
        print(f"Missing required columns in sensors CSV: {sorted(missing)}")
        sys.exit(1)

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
