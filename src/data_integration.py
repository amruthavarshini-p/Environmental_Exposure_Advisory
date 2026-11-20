import pandas as pd
import os


# ============================================================
# FILE PATHS
# ============================================================

OPENAQ_FILE = "data/raw/bengaluru_environmental_2025.csv"

WEATHER_FILE = "data/raw/bengaluru_weather_2025.csv"

UV_FILE = "data/raw/bengaluru_uv_2025.csv"

OUTPUT_FILE = "data/processed/bengaluru_integrated_2025.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    air = pd.read_csv(OPENAQ_FILE)

    weather = pd.read_csv(WEATHER_FILE)

    uv = pd.read_csv(UV_FILE)

    # Convert timestamps to UTC

    air["datetime"] = pd.to_datetime(
        air["datetime"],
        utc=True
    )

    weather["datetime"] = pd.to_datetime(
        weather["datetime"],
        utc=True
    )

    uv["datetime"] = pd.to_datetime(
        uv["datetime"],
        utc=True
    )

    print(f"\nOpenAQ shape: {air.shape}")

    print(f"Weather shape: {weather.shape}")

    print(f"UV shape: {uv.shape}")

    return air, weather, uv


# ============================================================
# PREPARE AIR DATA
# ============================================================

def prepare_air_data(air):

    print("\n")
    print("=" * 60)
    print("PREPARING OPENAQ DATA")
    print("=" * 60)

    # Open-Meteo provides the weather variables.
    # Remove the corresponding OpenAQ weather columns.

    weather_columns = [
        "temperature",
        "relativehumidity",
        "wind_speed",
        "wind_direction"
    ]

    existing_columns = [
        column
        for column in weather_columns
        if column in air.columns
    ]

    air = air.drop(
        columns=existing_columns
    )

    print("\nRemoved OpenAQ weather columns:")

    for column in existing_columns:

        print(f"  - {column}")

    return air


# ============================================================
# ALIGN OPENAQ TIMESTAMP
# ============================================================

def align_air_timestamp(air):

    print("\n")
    print("=" * 60)
    print("ALIGNING OPENAQ TIMESTAMPS")
    print("=" * 60)

    # OpenAQ timestamps are approximately :30.
    # Open-Meteo timestamps are at :00.
    #
    # Example:
    #
    # OpenAQ:
    # 2025-02-19 00:30
    #
    # Open-Meteo:
    # 2025-02-19 00:00
    #
    # Shift OpenAQ back by 30 minutes.

    air["datetime"] = (
        air["datetime"]
        - pd.Timedelta(minutes=30)
    )

    print(
        "\nOpenAQ timestamps shifted back by 30 minutes."
    )

    return air


# ============================================================
# PREPARE WEATHER DATA
# ============================================================

def prepare_weather_data(weather):

    # Remove UV from the weather dataset if it exists.
    #
    # We now have a dedicated UV dataset.

    if "uv_index" in weather.columns:

        weather = weather.drop(
            columns=["uv_index"]
        )

    return weather


# ============================================================
# MERGE AIR + WEATHER
# ============================================================

def merge_air_weather(air, weather):

    print("\n")
    print("=" * 60)
    print("MERGING OPENAQ + OPEN-METEO WEATHER")
    print("=" * 60)

    merged = pd.merge(
        air,
        weather,
        on=[
            "location",
            "datetime"
        ],
        how="left"
    )

    return merged


# ============================================================
# MERGE UV
# ============================================================

def merge_uv(merged, uv):

    print("\n")
    print("=" * 60)
    print("MERGING UV INDEX")
    print("=" * 60)

    # UV timestamps are already hourly at :00,
    # matching the aligned timestamps.

    merged = pd.merge(
        merged,
        uv,
        on=[
            "location",
            "datetime"
        ],
        how="left"
    )

    return merged


# ============================================================
# QUALITY REPORT
# ============================================================

def quality_report(df):

    print("\n")
    print("=" * 60)
    print("FINAL INTEGRATED DATASET QUALITY")
    print("=" * 60)

    print("\nShape:")

    print(df.shape)

    print("\nRows per location:")

    print(
        df.groupby(
            "location"
        ).size()
    )

    print("\nDate range:")

    print(
        "Start:",
        df["datetime"].min()
    )

    print(
        "End:",
        df["datetime"].max()
    )

    print("\nColumns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    print("\nMissing values:")

    print(
        df.isnull().sum()
    )

    print("\nMissing percentage:")

    print(
        (
            df.isnull().mean() * 100
        ).round(2)
    )

    print(
        "\nDuplicate location + timestamp:"
    )

    print(
        df.duplicated(
            subset=[
                "location",
                "datetime"
            ]
        ).sum()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Create processed directory

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    air, weather, uv = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    air = prepare_air_data(
        air
    )

    weather = prepare_weather_data(
        weather
    )

    # --------------------------------------------------------
    # Align OpenAQ timestamps
    # --------------------------------------------------------

    air = align_air_timestamp(
        air
    )

    # --------------------------------------------------------
    # Merge air + weather
    # --------------------------------------------------------

    merged = merge_air_weather(
        air,
        weather
    )

    # --------------------------------------------------------
    # Merge UV
    # --------------------------------------------------------

    merged = merge_uv(
        merged,
        uv
    )

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    merged = merged.drop_duplicates(
        subset=[
            "location",
            "datetime"
        ]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    merged = merged.sort_values(
        [
            "location",
            "datetime"
        ]
    )

    # --------------------------------------------------------
    # Quality report
    # --------------------------------------------------------

    quality_report(
        merged
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n")
    print("=" * 60)
    print("INTEGRATION COMPLETED")
    print("=" * 60)

    print(
        "\nFinal integrated dataset:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()