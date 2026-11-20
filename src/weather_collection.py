import os
import requests
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "data/raw"

START_DATE = "2025-02-19"
END_DATE = "2025-12-31"

LOCATIONS = {
    "Peenya": {
        "latitude": 13.0285,
        "longitude": 77.5199
    },

    "BTM Layout": {
        "latitude": 12.9166,
        "longitude": 77.6101
    },

    "Jayanagar 5th Block": {
        "latitude": 12.9250,
        "longitude": 77.5938
    },

    "Silk Board": {
        "latitude": 12.9170,
        "longitude": 77.6227
    }
}


# ============================================================
# OPEN-METEO HISTORICAL API
# ============================================================

API_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# WEATHER VARIABLES
# ============================================================

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "uv_index"
]


# ============================================================
# DOWNLOAD WEATHER DATA
# ============================================================

def download_weather(
    location_name,
    latitude,
    longitude
):

    print("\n" + "-" * 60)

    print(
        f"Downloading weather: {location_name}"
    )

    print("-" * 60)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC"
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    df.rename(
        columns={
            "time": "datetime",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "relativehumidity",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
            "uv_index": "uv_index"
        },
        inplace=True
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    df["location"] = location_name

    # Reorder columns
    columns = [
        "datetime",
        "temperature",
        "relativehumidity",
        "wind_speed",
        "wind_direction",
        "uv_index",
        "location"
    ]

    df = df[columns]

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\n")
    print("=" * 60)
    print("OPEN-METEO WEATHER DATA COLLECTION")
    print("=" * 60)

    print(
        f"\nPeriod: {START_DATE} → {END_DATE}"
    )

    all_weather = []

    # --------------------------------------------------------
    # Download each location
    # --------------------------------------------------------

    for location_name, coordinates in LOCATIONS.items():

        try:

            df = download_weather(
                location_name,
                coordinates["latitude"],
                coordinates["longitude"]
            )

            print(
                f"Rows downloaded: {len(df)}"
            )

            print(
                f"Columns: {len(df.columns)}"
            )

            # Save individual location data

            filename = (
                f"{OUTPUT_DIR}/"
                f"{location_name.lower().replace(' ', '_')}"
                f"_weather_2025.csv"
            )

            df.to_csv(
                filename,
                index=False
            )

            print(
                f"Saved: {filename}"
            )

            all_weather.append(df)

        except Exception as error:

            print(
                f"\nERROR for {location_name}:"
            )

            print(error)


    # --------------------------------------------------------
    # Combine all locations
    # --------------------------------------------------------

    if all_weather:

        combined = pd.concat(
            all_weather,
            ignore_index=True
        )

        combined = combined.sort_values(
            [
                "location",
                "datetime"
            ]
        )

        combined_file = (
            f"{OUTPUT_DIR}/"
            "bengaluru_weather_2025.csv"
        )

        combined.to_csv(
            combined_file,
            index=False
        )

        print("\n")
        print("=" * 60)
        print("WEATHER DATASET CREATED")
        print("=" * 60)

        print(
            f"\nFile: {combined_file}"
        )

        print(
            f"\nShape: {combined.shape}"
        )

        print(
            "\nRows per location:"
        )

        print(
            combined.groupby(
                "location"
            ).size()
        )

        print(
            "\nMissing values:"
        )

        print(
            combined.isnull().sum()
        )

        print(
            "\nDate range:"
        )

        print(
            "Start:",
            combined["datetime"].min()
        )

        print(
            "End:",
            combined["datetime"].max()
        )

        print("\n")
        print("=" * 60)
        print("WEATHER COLLECTION COMPLETED")
        print("=" * 60)

    else:

        print(
            "\nNo weather data was downloaded."
        )


if __name__ == "__main__":
    main()