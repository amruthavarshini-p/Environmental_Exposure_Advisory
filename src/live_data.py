import os
import requests
import pandas as pd
from dotenv import load_dotenv


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(
    os.path.join(PROJECT_ROOT, ".env")
)

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")


# ==========================================================
# BENGALURU STATIONS
# ==========================================================

STATIONS = {
    "Peenya": {
        "latitude": 13.0285,
        "longitude": 77.5197,
        "pm25_sensor": 12235370
    },

    "BTM Layout": {
        "latitude": 12.9166,
        "longitude": 77.6101,
        "pm25_sensor": 12235361
    },

    "Jayanagar 5th Block": {
        "latitude": 12.9250,
        "longitude": 77.5938,
        "pm25_sensor": 12235267
    },

    "Silk Board": {
        "latitude": 12.9172,
        "longitude": 77.6228,
        "pm25_sensor": 12235285
    }
}


# ==========================================================
# OPENAQ PM2.5
# ==========================================================

def get_recent_pm25(
    sensor_id,
    days=30
):

    if not OPENAQ_API_KEY:

        raise ValueError(
            "OPENAQ_API_KEY was not found in .env"
        )

    url = (
        f"https://api.openaq.org/v3/"
        f"sensors/{sensor_id}/hours"
    )

    now = pd.Timestamp.now(tz="UTC")

    datetime_from = (
        now - pd.Timedelta(days=days)
    ).isoformat()

    datetime_to = now.isoformat()

    response = requests.get(
        url,
        headers={
            "X-API-Key": OPENAQ_API_KEY
        },
        params={
            "datetime_from": datetime_from,
            "datetime_to": datetime_to,
            "limit": 1000,
            "page": 1
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    rows = []

    for item in data.get("results", []):

        period = item.get("period", {})

        datetime_from_info = period.get(
            "datetimeFrom",
            {}
        )

        timestamp = datetime_from_info.get(
            "utc"
        )

        value = item.get("value")

        if (
            timestamp is not None
            and value is not None
        ):

            rows.append(
                {
                    "datetime": timestamp,
                    "pm25": value
                }
            )

    if not rows:

        raise ValueError(
            "No recent PM2.5 observations were "
            "returned from OpenAQ."
        )

    df = pd.DataFrame(rows)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    df = (
        df.sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"]
        )
        .reset_index(drop=True)
    )

    return df


# ==========================================================
# OPEN-METEO WEATHER
# ==========================================================

def get_weather_for_period(
    latitude,
    longitude,
    start_time,
    end_time
):

    url = (
        "https://archive-api.open-meteo.com/"
        "v1/archive"
    )

    start_date = (
        pd.Timestamp(start_time)
        .strftime("%Y-%m-%d")
    )

    end_date = (
        pd.Timestamp(end_time)
        .strftime("%Y-%m-%d")
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "start_date": start_date,
        "end_date": end_date,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "wind_direction_10m"
        ),

        "timezone": "UTC"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:

        raise ValueError(
            "Open-Meteo did not return hourly weather data."
        )

    hourly = data["hourly"]

    weather_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                hourly["time"],
                utc=True
            ),

            "temperature": hourly[
                "temperature_2m"
            ],

            "relativehumidity": hourly[
                "relative_humidity_2m"
            ],

            "wind_speed": hourly[
                "wind_speed_10m"
            ],

            "wind_direction": hourly[
                "wind_direction_10m"
            ]
        }
    )

    return weather_df


# ==========================================================
# COMBINE LIVE ENVIRONMENTAL DATA
# ==========================================================

def get_live_environment(
    location="Peenya",
    days=30
):

    if location not in STATIONS:

        raise ValueError(
            f"Unknown location: {location}"
        )

    station = STATIONS[
        location
    ]

    # ------------------------------------------------------
    # Get recent PM2.5
    # ------------------------------------------------------

    pm25_df = get_recent_pm25(
        sensor_id=station["pm25_sensor"],
        days=days
    )

    # ------------------------------------------------------
    # Find actual PM2.5 time range
    # ------------------------------------------------------

    pm25_start = pm25_df[
        "datetime"
    ].min()

    pm25_end = pm25_df[
        "datetime"
    ].max()

    print(
        f"\nOpenAQ PM2.5 range:"
    )

    print(
        f"{pm25_start} → {pm25_end}"
    )

    # ------------------------------------------------------
    # Get weather for SAME period
    # ------------------------------------------------------

    weather_df = get_weather_for_period(
        latitude=station["latitude"],
        longitude=station["longitude"],
        start_time=pm25_start,
        end_time=pm25_end
    )

    # ------------------------------------------------------
    # Merge nearest timestamps
    # ------------------------------------------------------

    df = pd.merge_asof(
        pm25_df.sort_values("datetime"),
        weather_df.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta(
            "30min"
        )
    )

    df["location"] = location

    return df


# ==========================================================
# CREATE 24-HOUR SEQUENCE
# ==========================================================

def create_live_sequence(
    location="Peenya"
):

    df = get_live_environment(
        location=location,
        days=30
    )

    required_columns = [
        "pm25",
        "temperature",
        "relativehumidity",
        "wind_speed",
        "wind_direction"
    ]

    # ------------------------------------------------------
    # Check required columns
    # ------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
        )

    # ------------------------------------------------------
    # Remove rows where weather is unavailable
    # ------------------------------------------------------

    df = df.dropna(
        subset=[
            "temperature",
            "relativehumidity",
            "wind_speed",
            "wind_direction"
        ]
    )

    # ------------------------------------------------------
    # Keep only complete PM2.5 observations
    # ------------------------------------------------------

    df = df.dropna(
        subset=["pm25"]
    )

    # ------------------------------------------------------
    # Sort
    # ------------------------------------------------------

    df = df.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    # ------------------------------------------------------
    # Need at least 24 observations
    # ------------------------------------------------------

    if len(df) < 24:

        raise ValueError(
            f"Only {len(df)} usable hourly observations "
            "are available. At least 24 are required."
        )

    # ------------------------------------------------------
    # Take latest 24 observations
    # ------------------------------------------------------

    df = df.tail(24)

    return df


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "RECENT LIVE ENVIRONMENTAL DATA TEST"
    )
    print("=" * 60)

    try:

        df = create_live_sequence(
            location="Peenya"
        )

        print(
            "\nLive data retrieved successfully."
        )

        print("\nShape:")
        print(df.shape)

        print("\nColumns:")
        print(
            df.columns.tolist()
        )

        print("\nFinal time range:")

        print(
            df["datetime"].min(),
            "→",
            df["datetime"].max()
        )

        print("\nLatest 5 observations:")

        print(
            df[
                [
                    "datetime",
                    "pm25",
                    "temperature",
                    "relativehumidity",
                    "wind_speed",
                    "wind_direction"
                ]
            ].tail()
        )

        print("\nMissing values:")

        print(
            df[
                [
                    "pm25",
                    "temperature",
                    "relativehumidity",
                    "wind_speed",
                    "wind_direction"
                ]
            ].isnull().sum()
        )

        print("\nLatest PM2.5:")

        print(
            df["pm25"].iloc[-1]
        )

        print(
            "\nTEST SUCCESSFUL"
        )

    except Exception as e:

        print(
            "\nTEST FAILED"
        )

        print(
            type(e).__name__,
            ":",
            str(e)
        )