import pickle
import numpy as np
import tensorflow as tf


MODEL_PATH = "models/gru_best.keras"
TARGET_SCALER_PATH = "data/processed/target_scaler.pkl"

SEQUENCE_LENGTH = 24

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


# Load GRU
model = tf.keras.models.load_model(
    MODEL_PATH
)

# Load target scaler
with open(
    TARGET_SCALER_PATH,
    "rb"
) as file:

    target_scaler = pickle.load(file)


def predict_pm25(environmental_sequence):

    sequence = np.asarray(
        environmental_sequence,
        dtype=np.float32
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        len(FEATURES)
    )

    if sequence.shape != expected_shape:

        raise ValueError(
            f"Expected shape {expected_shape}, "
            f"received {sequence.shape}"
        )

    # Add batch dimension
    sequence = np.expand_dims(
        sequence,
        axis=0
    )

    # GRU prediction
    predicted_scaled = model.predict(
        sequence,
        verbose=0
    )

    # Convert back to µg/m³
    predicted_pm25 = (
        target_scaler.inverse_transform(
            predicted_scaled
        )
    )

    return float(
        predicted_pm25[0][0]
    )


# ============================================================
# TEST WITH REAL UNSEEN TEST DATA
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING REAL GRU PREDICTION")
    print("=" * 60)

    X_test = np.load(
        "data/processed/X_test.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    # Use the same first test sequence
    sample_sequence = X_test[0]

    actual_pm25 = (
        target_scaler.inverse_transform(
            y_test[0].reshape(1, 1)
        )[0][0]
    )

    predicted_pm25 = predict_pm25(
        sample_sequence
    )

    error = abs(
        actual_pm25 -
        predicted_pm25
    )

    print(
        f"\nActual PM2.5:"
    )

    print(
        f"{actual_pm25:.2f} µg/m³"
    )

    print(
        f"\nGRU predicted PM2.5:"
    )

    print(
        f"{predicted_pm25:.2f} µg/m³"
    )

    print(
        f"\nAbsolute error:"
    )

    print(
        f"{error:.2f} µg/m³"
    )

    print("\n") 
    print("=" * 60)
    print("GRU PREDICTION COMPLETED")
    print("=" * 60)