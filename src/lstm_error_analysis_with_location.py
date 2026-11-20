import numpy as np
import pandas as pd

from preprocessing import (
    load_data,
    resample_hourly,
    handle_missing_values,
    create_splits,
    scale_data
)

import pickle
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
MODEL_FILE = "models/lstm_pm25_best.keras"

SEQUENCE_LENGTH = 24

TARGET = "pm25"

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
    "wind_direction",
    "uv_index"
]


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

print("=" * 60)
print("LSTM LOCATION + TIMESTAMP ERROR ANALYSIS")
print("=" * 60)

print("\nLoading LSTM model...")

model = load_model(
    MODEL_FILE
)

with open(
    f"{DATA_DIR}/feature_scaler.pkl",
    "rb"
) as file:

    feature_scaler = pickle.load(
        file
    )

with open(
    f"{DATA_DIR}/target_scaler.pkl",
    "rb"
) as file:

    target_scaler = pickle.load(
        file
    )


# ============================================================
# LOAD AND RECREATE TEST DATA
# ============================================================

print("\nRecreating test metadata...")

df = load_data()

df = resample_hourly(
    df
)

df = handle_missing_values(
    df
)

(
    train_df,
    val_df,
    test_df
) = create_splits(
    df
)

(
    train_df,
    val_df,
    test_df
) = scale_data(
    train_df,
    val_df,
    test_df
)


# ============================================================
# CREATE TEST METADATA
# ============================================================

metadata = []

for location in test_df["location"].unique():

    location_df = test_df[
        test_df["location"] == location
    ].copy()

    location_df = location_df.sort_values(
        "datetime"
    )

    feature_values = location_df[
        FEATURES
    ].values

    target_values = location_df[
        TARGET
    ].values

    datetime_values = location_df[
        "datetime"
    ].values

    for i in range(
        SEQUENCE_LENGTH,
        len(location_df)
    ):

        # Same target filtering used during preprocessing

        if np.isnan(
            target_values[i]
        ):

            continue

        sequence = feature_values[
            i - SEQUENCE_LENGTH:i
        ]

        # Same sequence validation used during preprocessing

        if np.isnan(
            sequence
        ).any():

            continue

        metadata.append({

            "datetime": datetime_values[i],

            "location": location,

            "actual_scaled": target_values[i]

        })


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = np.load(
    f"{DATA_DIR}/X_test.npy"
)

y_test = np.load(
    f"{DATA_DIR}/y_test.npy"
)


print(
    f"\nTest sequences loaded: {len(X_test)}"
)

print(
    f"Metadata rows created: {len(metadata)}"
)


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating LSTM predictions...")

predicted_scaled = model.predict(
    X_test,
    verbose=0
).flatten()


# ============================================================
# CONVERT BACK TO ORIGINAL PM2.5
# ============================================================

def inverse_pm25_scaling(values):

    values = np.asarray(
        values
    ).reshape(-1, 1)

    # Reverse target scaling

    values = target_scaler.inverse_transform(
        values
    )

    # Reverse feature scaling for PM2.5
    # PM2.5 is feature index 0

    original_min = (
        feature_scaler.data_min_[0]
    )

    original_max = (
        feature_scaler.data_max_[0]
    )

    values = (
        values *
        (
            original_max -
            original_min
        )
        +
        original_min
    )

    return values.flatten()


actual_pm25 = inverse_pm25_scaling(
    y_test
)

predicted_pm25 = inverse_pm25_scaling(
    predicted_scaled
)


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

metadata_df = pd.DataFrame(
    metadata
)

result = pd.DataFrame({

    "datetime":
        metadata_df["datetime"],

    "location":
        metadata_df["location"],

    "actual_pm25_ug_m3":
        actual_pm25,

    "predicted_pm25_ug_m3":
        predicted_pm25

})

result["absolute_error"] = np.abs(
    result["actual_pm25_ug_m3"] -
    result["predicted_pm25_ug_m3"]
)


# ============================================================
# SORT BY ERROR
# ============================================================

result = result.sort_values(
    "absolute_error",
    ascending=False
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    f"{DATA_DIR}/"
    "lstm_predictions_with_location.csv"
)

result.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY TOP ERRORS
# ============================================================

print("\n")
print("=" * 60)
print("TOP 20 LSTM PREDICTION ERRORS")
print("=" * 60)

print(
    result.head(20).to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("ERROR SUMMARY")
print("=" * 60)

print(
    f"\nTotal test predictions: "
    f"{len(result)}"
)

print(
    f"\nMedian error: "
    f"{result['absolute_error'].median():.4f} µg/m³"
)

print(
    f"\n90th percentile error: "
    f"{result['absolute_error'].quantile(0.90):.4f} µg/m³"
)

print(
    f"\n95th percentile error: "
    f"{result['absolute_error'].quantile(0.95):.4f} µg/m³"
)

print(
    f"\nMaximum error: "
    f"{result['absolute_error'].max():.4f} µg/m³"
)

print("\n")
print("=" * 60)
print("FILE SAVED")
print("=" * 60)

print(
    f"\n{output_file}"
)

print("\nAnalysis completed.")