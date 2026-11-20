import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/bengaluru_integrated_2025.csv"
OUTPUT_DIR = "data/processed"

SEQUENCE_LENGTH = 24
TARGET = "pm25"

# PM2.5 is included as an input feature because the model
# uses previous PM2.5 values to predict the next PM2.5 value.
#
# UV is temporarily excluded because the current integrated
# dataset contains 100% missing UV values.
FEATURES = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co",
    "temperature",
    "relativehumidity",
    "wind_speed",
    "wind_direction"
]

INPUT_FEATURES = [
    feature
    for feature in FEATURES
    if feature != TARGET
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING INTEGRATED DATASET")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    df = df.sort_values(
        ["location", "datetime"]
    ).reset_index(drop=True)

    print("\nDataset shape:")
    print(df.shape)

    print("\nLocations:")
    print(df["location"].unique())

    print("\nDate range:")
    print(
        df["datetime"].min(),
        "to",
        df["datetime"].max()
    )

    return df


# ============================================================
# RESAMPLE HOURLY
# ============================================================

def resample_hourly(df):

    print("\n")
    print("=" * 60)
    print("RESAMPLING DATA TO HOURLY FREQUENCY")
    print("=" * 60)

    all_locations = []

    for location in df["location"].unique():

        print(f"\nProcessing: {location}")

        location_df = df[
            df["location"] == location
        ].copy()

        location_df = location_df.set_index(
            "datetime"
        )

        location_df = location_df[
            FEATURES
        ]

        location_df = location_df.resample(
            "1h"
        ).mean()

        location_df["location"] = location

        location_df = location_df.reset_index()

        all_locations.append(
            location_df
        )

    result = pd.concat(
        all_locations,
        ignore_index=True
    )

    result = result.sort_values(
        ["location", "datetime"]
    ).reset_index(drop=True)

    print(
        f"\nResampled shape: {result.shape}"
    )

    return result


# ============================================================
# CREATE TIME-BASED SPLITS
# ============================================================

def create_splits(df):

    print("\n")
    print("=" * 60)
    print("CREATING TIME-BASED DATA SPLITS")
    print("=" * 60)

    train_parts = []
    val_parts = []
    test_parts = []

    for location in df["location"].unique():

        location_df = df[
            df["location"] == location
        ].copy()

        location_df = location_df.sort_values(
            "datetime"
        ).reset_index(drop=True)

        n = len(location_df)

        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_location = location_df.iloc[
            :train_end
        ].copy()

        val_location = location_df.iloc[
            train_end:val_end
        ].copy()

        test_location = location_df.iloc[
            val_end:
        ].copy()

        train_parts.append(
            train_location
        )

        val_parts.append(
            val_location
        )

        test_parts.append(
            test_location
        )

        print(f"\n{location}")
        print("Total:", n)
        print("Train:", len(train_location))
        print("Validation:", len(val_location))
        print("Test:", len(test_location))

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    val_df = pd.concat(
        val_parts,
        ignore_index=True
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True
    )

    print("\nTrain shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(
    train_df,
    val_df,
    test_df
):

    print("\n")
    print("=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)

    print(
        "\nPM2.5 will NOT be imputed because it is the target."
    )

    print(
        "\nInput features:"
    )

    print(
        INPUT_FEATURES
    )

    # --------------------------------------------------------
    # Calculate fallback medians ONLY from training data.
    # --------------------------------------------------------

    training_medians = {}

    for column in INPUT_FEATURES:

        training_medians[column] = (
            train_df[column].median()
        )

    # --------------------------------------------------------
    # Fill input features using FORWARD FILL only.
    #
    # This prevents using future observations to predict
    # an earlier observation.
    #
    # Remaining missing values use TRAINING-ONLY medians.
    # --------------------------------------------------------

    for dataframe in [
        train_df,
        val_df,
        test_df
    ]:

        for column in INPUT_FEATURES:

            dataframe[column] = (
                dataframe
                .groupby("location")[column]
                .ffill()
            )

            dataframe[column] = (
                dataframe[column]
                .fillna(
                    training_medians[column]
                )
            )

    # --------------------------------------------------------
    # Report missing values.
    # --------------------------------------------------------

    print(
        "\nMissing values AFTER processing:"
    )

    print("\nTraining:")

    print(
        train_df[
            FEATURES
        ].isnull().sum()
    )

    print("\nValidation:")

    print(
        val_df[
            FEATURES
        ].isnull().sum()
    )

    print("\nTesting:")

    print(
        test_df[
            FEATURES
        ].isnull().sum()
    )

    print(
        "\nPM2.5 missing values are intentionally preserved."
    )

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# SCALE DATA
# ============================================================

def scale_data(
    train_df,
    val_df,
    test_df
):

    print("\n")
    print("=" * 60)
    print("SCALING FEATURES AND TARGET")
    print("=" * 60)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # PM2.5 has TWO roles:
    #
    # 1. Previous PM2.5 = input feature
    # 2. Future PM2.5 = prediction target
    #
    # Therefore PM2.5 gets its OWN scaler.
    #
    # Other environmental variables get a separate
    # feature scaler.
    # --------------------------------------------------------

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # --------------------------------------------------------
    # Fit feature scaler ONLY on training input features.
    # --------------------------------------------------------

    train_df[INPUT_FEATURES] = (
        feature_scaler.fit_transform(
            train_df[INPUT_FEATURES]
        )
    )

    val_df[INPUT_FEATURES] = (
        feature_scaler.transform(
            val_df[INPUT_FEATURES]
        )
    )

    test_df[INPUT_FEATURES] = (
        feature_scaler.transform(
            test_df[INPUT_FEATURES]
        )
    )

    # --------------------------------------------------------
    # Fit target scaler ONLY on raw training PM2.5.
    #
    # IMPORTANT:
    # This happens BEFORE PM2.5 is scaled as an input feature.
    # --------------------------------------------------------

    train_target_raw = (
        train_df[
            [TARGET]
        ].dropna()
    )

    target_scaler.fit(
        train_target_raw
    )

    # --------------------------------------------------------
    # Scale PM2.5 separately.
    #
    # Missing PM2.5 remains NaN.
    # --------------------------------------------------------

    for dataframe in [
        train_df,
        val_df,
        test_df
    ]:

        valid_mask = (
            dataframe[TARGET].notna()
        )

        dataframe.loc[
            valid_mask,
            TARGET
        ] = target_scaler.transform(
            dataframe.loc[
                valid_mask,
                [TARGET]
            ]
        ).flatten()

    # --------------------------------------------------------
    # Save scalers.
    # --------------------------------------------------------

    with open(
        f"{OUTPUT_DIR}/feature_scaler.pkl",
        "wb"
    ) as file:

        pickle.dump(
            feature_scaler,
            file
        )

    with open(
        f"{OUTPUT_DIR}/target_scaler.pkl",
        "wb"
    ) as file:

        pickle.dump(
            target_scaler,
            file
        )

    print(
        "\nFeature scaler:"
    )

    print(
        "Fitted ONLY on training input features."
    )

    print(
        "\nTarget scaler:"
    )

    print(
        "Fitted ONLY on raw training PM2.5."
    )

    print(
        "\nSaved:"
    )

    print(
        f"{OUTPUT_DIR}/feature_scaler.pkl"
    )

    print(
        f"{OUTPUT_DIR}/target_scaler.pkl"
    )

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(df):

    X = []
    y = []

    for location in df["location"].unique():

        location_df = df[
            df["location"] == location
        ].copy()

        location_df = location_df.sort_values(
            "datetime"
        ).reset_index(drop=True)

        feature_values = (
            location_df[
                FEATURES
            ].values
        )

        target_values = (
            location_df[
                TARGET
            ].values
        )

        timestamps = (
            location_df[
                "datetime"
            ].values
        )

        for i in range(
            SEQUENCE_LENGTH,
            len(location_df)
        ):

            # ------------------------------------------------
            # Target must exist.
            # ------------------------------------------------

            if np.isnan(
                target_values[i]
            ):

                continue

            # ------------------------------------------------
            # Make sure the previous 24 observations are
            # actually consecutive hourly observations.
            # ------------------------------------------------

            sequence_times = timestamps[
                i - SEQUENCE_LENGTH:i
            ]

            time_differences = (
                np.diff(
                    sequence_times
                )
            )

            expected_difference = (
                np.timedelta64(
                    1,
                    "h"
                )
            )

            if not np.all(
                time_differences ==
                expected_difference
            ):

                continue

            # ------------------------------------------------
            # Previous 24 hours.
            # ------------------------------------------------

            sequence = feature_values[
                i - SEQUENCE_LENGTH:i
            ]

            # ------------------------------------------------
            # No missing input values.
            # ------------------------------------------------

            if np.isnan(
                sequence
            ).any():

                continue

            X.append(
                sequence
            )

            y.append(
                target_values[i]
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    print(
        "\nX shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    return (
        X,
        y
    )


# ============================================================
# SAVE DATA
# ============================================================

def save_data(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    np.save(
        f"{OUTPUT_DIR}/X_train.npy",
        X_train
    )

    np.save(
        f"{OUTPUT_DIR}/y_train.npy",
        y_train
    )

    np.save(
        f"{OUTPUT_DIR}/X_val.npy",
        X_val
    )

    np.save(
        f"{OUTPUT_DIR}/y_val.npy",
        y_val
    )

    np.save(
        f"{OUTPUT_DIR}/X_test.npy",
        X_test
    )

    np.save(
        f"{OUTPUT_DIR}/y_test.npy",
        y_test
    )

    print("\n")
    print("=" * 60)
    print("PROCESSED DATA SAVED")
    print("=" * 60)

    print(
        f"\nFiles saved in: {OUTPUT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # 1. Load
    df = load_data()

    # 2. Resample
    df = resample_hourly(df)

    # 3. Split BEFORE preprocessing
    (
        train_df,
        val_df,
        test_df
    ) = create_splits(df)

    # 4. Handle missing input features
    (
        train_df,
        val_df,
        test_df
    ) = handle_missing_values(
        train_df,
        val_df,
        test_df
    )

    # 5. Scale correctly
    (
        train_df,
        val_df,
        test_df
    ) = scale_data(
        train_df,
        val_df,
        test_df
    )

    # 6. Create sequences
    X_train, y_train = create_sequences(
        train_df
    )

    X_val, y_val = create_sequences(
        val_df
    )

    X_test, y_test = create_sequences(
        test_df
    )

    # 7. Save
    save_data(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(
        "\nTraining sequences:",
        X_train.shape
    )

    print(
        "Validation sequences:",
        X_val.shape
    )

    print(
        "Testing sequences:",
        X_test.shape
    )

    print(
        "\nSequence length:",
        SEQUENCE_LENGTH,
        "hours"
    )

    print(
        "Input features:",
        len(FEATURES)
    )

    print(
        "\nFeatures:"
    )

    for feature in FEATURES:
        print(
            "-",
            feature
        )

    print(
        "\nTarget:",
        TARGET
    )

    print(
        "\nLeakage prevention:"
    )

    print(
        "1. Time-based split performed before preprocessing."
    )

    print(
        "2. Input features use past-only forward filling."
    )

    print(
        "3. Remaining input missing values use training-only medians."
    )

    print(
        "4. Feature scaler fitted on training input features only."
    )

    print(
        "5. Target scaler fitted on raw training PM2.5 only."
    )

    print(
        "6. PM2.5 target values were never imputed."
    )

    print(
        "7. Sequences require consecutive hourly timestamps."
    )

    print(
        "8. UV index excluded until its missing data is fixed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()