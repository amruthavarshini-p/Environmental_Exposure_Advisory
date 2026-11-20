import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAQ_API_KEY not found in .env file"
    )


# ============================================================
# OPENAQ CONFIGURATION
# ============================================================

BASE_URL = "https://api.openaq.org/v3"

HEADERS = {
    "X-API-Key": API_KEY
}


# ============================================================
# BENGALURU LOCATIONS
# ============================================================

LOCATIONS = {
    "Peenya": 5607,
    "BTM Layout": 5548,
    "Jayanagar 5th Block": 6973,
    "Silk Board": 6975,
}


# ============================================================
# PARAMETERS REQUIRED FOR THE PROJECT
# ============================================================

PARAMETERS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co",
    "temperature",
    "relativehumidity",
    "wind_speed",
    "wind_direction",
]


# ============================================================
# DATA COLLECTION PERIOD
# ============================================================

START_DATE = "2025-02-19T00:00:00Z"
END_DATE = "2025-12-31T23:59:59Z"


# ============================================================
# GET SENSORS FOR A LOCATION
# ============================================================

def get_sensors(location_id):

    url = f"{BASE_URL}/locations/{location_id}/sensors"

    params = {
        "limit": 100
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json().get("results", [])


# ============================================================
# SELECT SENSOR
# ============================================================

def select_sensors(sensors):

    selected = {}

    for sensor in sensors:

        parameter = sensor.get("parameter", {})

        parameter_name = parameter.get("name")

        if parameter_name not in PARAMETERS:
            continue

        sensor_id = sensor.get("id")

        last_date = sensor.get("datetimeLast")

        if sensor_id is None:
            continue

        # Select the sensor with the latest available data
        if (
            parameter_name not in selected
            or str(last_date) > str(
                selected[parameter_name]["datetimeLast"]
            )
        ):

            selected[parameter_name] = {
                "id": sensor_id,
                "datetimeLast": last_date,
                "units": parameter.get("units")
            }

    return selected


# ============================================================
# DOWNLOAD SENSOR DATA
# ============================================================

def download_sensor_data(sensor_id, parameter_name):

    url = f"{BASE_URL}/sensors/{sensor_id}/hours"

    all_rows = []

    start = pd.Timestamp(START_DATE)

    end = pd.Timestamp(END_DATE)

    current = start

    # --------------------------------------------------------
    # Download one week at a time
    # --------------------------------------------------------

    while current < end:

        chunk_end = min(
            current + pd.Timedelta(days=7),
            end
        )

        print(
            f"    Downloading "
            f"{current.strftime('%Y-%m-%d')} → "
            f"{chunk_end.strftime('%Y-%m-%d')}"
        )

        params = {
            "datetime_from": current.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),

            "datetime_to": chunk_end.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),

            "limit": 1000,

            "page": 1
        }

        success = False

        # ----------------------------------------------------
        # Retry failed requests
        # ----------------------------------------------------

        for attempt in range(3):

            try:

                response = requests.get(
                    url,
                    headers=HEADERS,
                    params=params,
                    timeout=60
                )

                response.raise_for_status()

                success = True

                break

            except requests.exceptions.HTTPError as error:

                print(
                    f"    API error "
                    f"(attempt {attempt + 1}/3): "
                    f"{error}"
                )

                if attempt < 2:

                    print(
                        "    Waiting 5 seconds before retry..."
                    )

                    time.sleep(5)

        # ----------------------------------------------------
        # Stop if all retries failed
        # ----------------------------------------------------

        if not success:

            raise RuntimeError(
                f"\nFailed to download {parameter_name} "
                f"for period "
                f"{current.date()} to {chunk_end.date()}."
            )

        # ----------------------------------------------------
        # Extract results
        # ----------------------------------------------------

        results = response.json().get("results", [])

        print(
            f"    Rows received: {len(results)}"
        )

        for row in results:

            period = row.get("period", {})

            value = row.get("value")

            datetime_value = (
                period
                .get("datetimeFrom", {})
                .get("utc")
            )

            if datetime_value is not None:

                all_rows.append({
                    "datetime": datetime_value,
                    parameter_name: value
                })

        # Move to next week

        current = chunk_end

        # Small pause between API requests

        time.sleep(0.5)

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(all_rows)

    if not df.empty:

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            utc=True
        )

        # Remove duplicate timestamps
        df = df.drop_duplicates(
            subset=["datetime"]
        )

        df = df.sort_values(
            "datetime"
        )

    return df


# ============================================================
# MAIN DATA COLLECTION
# ============================================================

def main():

    # Create raw data directory

    os.makedirs(
        "data/raw",
        exist_ok=True
    )

    all_location_data = []

    print("\n")
    print("=" * 60)
    print("OPENAQ FULL DATA COLLECTION")
    print("=" * 60)

    print("\nData period:")
    print(
        f"{START_DATE} → {END_DATE}"
    )

    print("\nLocations:")

    for location in LOCATIONS:

        print(
            f"  - {location}"
        )

    print("\nParameters:")

    for parameter in PARAMETERS:

        print(
            f"  - {parameter}"
        )


    # ========================================================
    # PROCESS EACH LOCATION
    # ========================================================

    for location_name, location_id in LOCATIONS.items():

        print("\n")
        print("-" * 60)
        print(
            f"LOCATION: {location_name}"
        )
        print(
            f"Location ID: {location_id}"
        )
        print("-" * 60)


        # ----------------------------------------------------
        # Find sensors
        # ----------------------------------------------------

        print("\nFinding available sensors...")

        try:

            sensors = get_sensors(
                location_id
            )

        except Exception as error:

            print(
                f"\nERROR while getting sensors:"
            )

            print(error)

            continue


        # ----------------------------------------------------
        # Select sensors
        # ----------------------------------------------------

        selected = select_sensors(
            sensors
        )


        print("\nSelected sensors:")

        for parameter in PARAMETERS:

            if parameter in selected:

                sensor = selected[
                    parameter
                ]

                print(
                    f"  {parameter:18} "
                    f"Sensor ID: {sensor['id']} "
                    f"Last: {sensor['datetimeLast']}"
                )

            else:

                print(
                    f"  {parameter:18} "
                    f"NOT AVAILABLE"
                )


        # ----------------------------------------------------
        # Download each parameter
        # ----------------------------------------------------

        location_data = None


        for parameter in PARAMETERS:

            if parameter not in selected:

                continue


            sensor_id = selected[
                parameter
            ]["id"]


            print("\n")
            print(
                f"Downloading {parameter}..."
            )

            try:

                df = download_sensor_data(
                    sensor_id,
                    parameter
                )

            except Exception as error:

                print(
                    f"\nERROR downloading "
                    f"{parameter}:"
                )

                print(error)

                print(
                    "\nSkipping this parameter..."
                )

                continue


            if df.empty:

                print(
                    f"  No data found for "
                    f"{parameter}"
                )

                continue


            print(
                f"  Total rows: {len(df)}"
            )


            # ------------------------------------------------
            # Merge parameter with location dataset
            # ------------------------------------------------

            if location_data is None:

                location_data = df

            else:

                location_data = pd.merge(
                    location_data,
                    df,
                    on="datetime",
                    how="outer"
                )


        # ----------------------------------------------------
        # Check location data
        # ----------------------------------------------------

        if (
            location_data is None
            or location_data.empty
        ):

            print(
                f"\nNo usable data found "
                f"for {location_name}."
            )

            continue


        # ----------------------------------------------------
        # Add location
        # ----------------------------------------------------

        location_data["location"] = (
            location_name
        )


        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        location_data = location_data.sort_values(
            "datetime"
        )


        # ----------------------------------------------------
        # Save individual location dataset
        # ----------------------------------------------------

        filename = (
            "data/raw/"
            + location_name.lower()
            .replace(" ", "_")
            + "_2025.csv"
        )


        location_data.to_csv(
            filename,
            index=False
        )


        print("\n")
        print(
            f"Location dataset saved:"
        )

        print(filename)

        print(
            f"Rows: {len(location_data)}"
        )

        print(
            f"Columns: "
            f"{len(location_data.columns)}"
        )


        # Add to combined dataset

        all_location_data.append(
            location_data
        )


    # ========================================================
    # COMBINE ALL LOCATIONS
    # ========================================================

    if all_location_data:

        print("\n")
        print("=" * 60)
        print("COMBINING BENGALURU DATASETS")
        print("=" * 60)


        combined = pd.concat(
            all_location_data,
            ignore_index=True
        )


        # Sort by location and datetime

        combined = combined.sort_values(
            [
                "location",
                "datetime"
            ]
        )


        # Remove exact duplicates

        combined = combined.drop_duplicates(
            subset=[
                "location",
                "datetime"
            ]
        )


        # Save combined dataset

        combined_file = (
            "data/raw/"
            "bengaluru_environmental_2025.csv"
        )


        combined.to_csv(
            combined_file,
            index=False
        )


        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        print("\n")
        print("=" * 60)
        print("FULL DATASET CREATED")
        print("=" * 60)


        print("\nFile:")
        print(
            combined_file
        )


        print("\nShape:")

        print(
            combined.shape
        )


        print("\nRows per location:")

        print(
            combined
            .groupby("location")
            .size()
        )


        print("\nColumns:")

        for column in combined.columns:

            print(
                f"  - {column}"
            )


        print("\nDate range:")

        print(
            "Start:",
            combined["datetime"].min()
        )

        print(
            "End:",
            combined["datetime"].max()
        )


        print("\nMissing values:")

        print(
            combined.isnull().sum()
        )


        print("\n")
        print("=" * 60)
        print("DATA COLLECTION COMPLETED")
        print("=" * 60)


    else:

        print("\n")
        print("=" * 60)
        print("NO DATA WAS DOWNLOADED")
        print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()