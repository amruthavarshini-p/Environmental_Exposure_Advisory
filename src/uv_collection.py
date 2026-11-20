import requests
import pandas as pd
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "data/raw"

START_DATE = "2025-02-19"
END_DATE = "2025-12-31"


# ============================================================
# BENGALURU LOCATIONS
# ============================================================

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
# OPEN-METEO HISTORICAL FORECAST API
# ============================================================

API_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)


# ============================================================
# DOWNLOAD UV DATA
# ============================================================

def download_uv(
    location_name,
    latitude,
    longitude
):

    print("\n" + "-" * 60)

    print(
        f"Downloading UV data: {location_name}"
    )

    print("-" * 60)

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "start_date": START_DATE,

        "end_date": END_DATE,

        "hourly": "uv_index",

        "timezone": "UTC"
    }


    response = requests.get(
        API_URL,
        params=params,
        timeout=120
    )


    response.raise_for_status()


    data = response.json()


    # --------------------------------------------------------
    # Check API response
    # --------------------------------------------------------

    if "hourly" not in data:

        raise ValueError(
            "Hourly UV data was not returned by Open-Meteo."
        )


    hourly = data["hourly"]


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame({

        "datetime": hourly["time"],

        "uv_index": hourly["uv_index"]

    })


    # --------------------------------------------------------
    # Convert datetime
    # --------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )


    # --------------------------------------------------------
    # Add location
    # --------------------------------------------------------

    df["location"] = location_name


    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    df = df[
        [
            "datetime",
            "uv_index",
            "location"
        ]
    ]


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

    print("OPEN-METEO UV INDEX COLLECTION")

    print("=" * 60)


    print(
        f"\nPeriod:"
    )

    print(
        f"{START_DATE} → {END_DATE}"
    )


    all_uv = []


    # ========================================================
    # DOWNLOAD EACH LOCATION
    # ========================================================

    for location_name, coordinates in LOCATIONS.items():

        try:

            df = download_uv(

                location_name,

                coordinates["latitude"],

                coordinates["longitude"]

            )


            print(
                f"Rows downloaded: {len(df)}"
            )


            print(
                f"Missing UV values: "
                f"{df['uv_index'].isna().sum()}"
            )


            # ------------------------------------------------
            # Save individual location file
            # ------------------------------------------------

            filename = (

                f"{OUTPUT_DIR}/"

                f"{location_name.lower().replace(' ', '_')}"

                f"_uv_2025.csv"

            )


            df.to_csv(
                filename,
                index=False
            )


            print(
                f"Saved: {filename}"
            )


            all_uv.append(df)


            # Small delay between requests

            time.sleep(1)


        except Exception as error:

            print(
                f"\nERROR for {location_name}:"
            )

            print(error)


    # ========================================================
    # COMBINE UV DATA
    # ========================================================

    if all_uv:

        combined = pd.concat(
            all_uv,
            ignore_index=True
        )


        combined = combined.sort_values(
            [
                "location",
                "datetime"
            ]
        )


        # Remove duplicates

        combined = combined.drop_duplicates(
            subset=[
                "location",
                "datetime"
            ]
        )


        # ----------------------------------------------------
        # Save combined UV dataset
        # ----------------------------------------------------

        output_file = (
            f"{OUTPUT_DIR}/"
            "bengaluru_uv_2025.csv"
        )


        combined.to_csv(
            output_file,
            index=False
        )


        # ====================================================
        # FINAL REPORT
        # ====================================================

        print("\n")

        print("=" * 60)

        print("UV DATASET CREATED")

        print("=" * 60)


        print(
            f"\nFile:"
        )

        print(
            output_file
        )


        print(
            f"\nShape:"
        )

        print(
            combined.shape
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
            "\nUV statistics:"
        )

        print(
            combined["uv_index"].describe()
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

        print("UV COLLECTION COMPLETED")

        print("=" * 60)


    else:

        print(
            "\nNo UV data was downloaded."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()