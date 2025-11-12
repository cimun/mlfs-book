#!/usr/bin/env python3
import sys
import os
import json
import re
import warnings
import subprocess
from pathlib import Path
from datetime import date
import datetime as dt

warnings.filterwarnings("ignore", module="IPython")

# -------- Environment / paths --------

root_dir = Path().absolute()
if root_dir.parts[-1:] == ("airquality",):
    root_dir = Path(*root_dir.parts[:-1])
if root_dir.parts[-1:] == ("notebooks",):
    root_dir = Path(*root_dir.parts[:-1])
print("Local environment")

root_dir = str(root_dir)
print(f"Root dir: {root_dir}")

if root_dir not in sys.path:
    sys.path.append(root_dir)
    print(f"Added to PYTHONPATH: {root_dir}")

# -------- Settings --------
from mlfs import config
settings = config.HopsworksSettings(_env_file=f"{root_dir}/.env")

# -------- Imports for pipeline --------
import pandas as pd
import numpy as np
import requests  # noqa: F401 (kept for parity with notebook)
import hopsworks
from mlfs.airquality import util
import great_expectations as ge

# -------- Hopsworks login --------
project = hopsworks.login(engine="python")
fs = project.get_feature_store()
secrets = hopsworks.get_secrets_api()

# -------- User/configured inputs --------
today = date.today()
csv_file = f"{root_dir}/data/schottenfeldgasse.csv"
util.check_file_path(csv_file)

if settings.AQICN_API_KEY is None:
    print("You need to set AQICN_API_KEY either in .env or here.")
    sys.exit(1)

AQICN_API_KEY = settings.AQICN_API_KEY.get_secret_value()
aqicn_url = settings.AQICN_URL
country = settings.AQICN_COUNTRY
city = settings.AQICN_CITY
street = settings.AQICN_STREET

# If needed, replace with util.get_city_coordinates(city)
latitude = "48.200458"
longitude = "16.343257"

print(f"Found AQICN_API_KEY: {AQICN_API_KEY}")

# Store AQICN_API_KEY as secret (replace if exists)
try:
    secret = secrets.get_secret("AQICN_API_KEY")
    if secret is not None:
        secret.delete()
        print("Replacing existing AQICN_API_KEY")
except Exception:
    pass
secrets.create_secret("AQICN_API_KEY", AQICN_API_KEY)

# -------- Validate API token quickly --------
try:
    _ = util.get_pm25(aqicn_url, country, city, street, today, AQICN_API_KEY)
    print("AQICN_API_KEY appears to work for the sensor.")
except hopsworks.RestAPIError:
    print("AQICN_API_KEY failed for your sensor. Check the key and sensor URL.")

# -------- Load and clean AQ CSV --------
df = pd.read_csv(csv_file, parse_dates=["date"], skipinitialspace=True)
df_aq = df[["date", "pm25"]].copy()
df_aq["pm25"] = df_aq["pm25"].astype("float32")
print(df_aq.info())

df_aq.dropna(inplace=True)
df_aq["country"] = country
df_aq["city"] = city
df_aq["street"] = street
df_aq["url"] = aqicn_url
print(df_aq.info())

# -------- Historical weather --------
earliest_aq_date = pd.Series.min(df_aq["date"]).strftime("%Y-%m-%d")
weather_df = util.get_historical_weather(city, earliest_aq_date, str(today), latitude, longitude)
print(weather_df.info())

# -------- Data validation (GE) --------
aq_expectation_suite = ge.core.ExpectationSuite(expectation_suite_name="aq_expectation_suite")
aq_expectation_suite.add_expectation(
    ge.core.ExpectationConfiguration(
        expectation_type="expect_column_min_to_be_between",
        kwargs={"column": "pm25", "min_value": -0.1, "max_value": 500.0, "strict_min": True},
    )
)

weather_expectation_suite = ge.core.ExpectationSuite(expectation_suite_name="weather_expectation_suite")
def expect_greater_than_zero(col: str):
    weather_expectation_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_min_to_be_between",
            kwargs={"column": col, "min_value": -0.1, "max_value": 1000.0, "strict_min": True},
        )
    )
expect_greater_than_zero("precipitation_sum")
expect_greater_than_zero("wind_speed_10m_max")

# -------- Save sensor location JSON as secret --------
location_dict = {
    "country": country,
    "city": city,
    "street": street,
    "aqicn_url": aqicn_url,
    "latitude": latitude,
    "longitude": longitude,
}
str_dict = json.dumps(location_dict)

try:
    secret = secrets.get_secret("SENSOR_LOCATION_JSON")
    if secret is not None:
        secret.delete()
        print("Replacing existing SENSOR_LOCATION_JSON")
except Exception:
    pass
secrets.create_secret("SENSOR_LOCATION_JSON", str_dict)

# -------- Feature Groups: Air Quality --------
air_quality_fg = fs.get_or_create_feature_group(
    name="air_quality",
    description="Air Quality characteristics of each day",
    version=1,
    primary_key=["country", "city", "street"],
    event_time="date",
    expectation_suite=aq_expectation_suite,
)

df_aq["pm25"] = df_aq["pm25"].astype(np.double)
print(df_aq.info())

air_quality_fg.insert(df_aq)
air_quality_fg.update_feature_description("date", "Date of measurement of air quality")
air_quality_fg.update_feature_description("country", "Country (sometimes a city in aqicn.org)")
air_quality_fg.update_feature_description("city", "City where the air quality was measured")
air_quality_fg.update_feature_description("street", "Street where the air quality was measured")
air_quality_fg.update_feature_description("pm25", "PM2.5 (μg/m3)")

# -------- Feature Groups: Weather --------
weather_fg = fs.get_or_create_feature_group(
    name="weather",
    description="Weather characteristics of each day",
    version=1,
    primary_key=["city"],
    event_time="date",
    expectation_suite=weather_expectation_suite,
)

weather_fg.insert(weather_df, wait=True)
weather_fg.update_feature_description("date", "Date of weather measurement")
weather_fg.update_feature_description("city", "City for weather")
weather_fg.update_feature_description("temperature_2m_mean", "Temperature in Celsius")
weather_fg.update_feature_description("precipitation_sum", "Precipitation (mm)")
weather_fg.update_feature_description("wind_speed_10m_max", "Max wind speed at 10m")
weather_fg.update_feature_description("wind_direction_10m_dominant", "Dominant wind direction")

print("Backfill complete.")
