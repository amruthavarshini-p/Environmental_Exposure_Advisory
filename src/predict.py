import pickle
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/lstm_best.keras"
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


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING LSTM MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("LSTM model loaded successfully.")


# ============================================================
# LOAD TARGET SCALER
# ============================================================

with open(
    TARGET_SCALER_PATH,
    "rb"
) as file:

    target_scaler = pickle.load(
        file
    )

print("Target scaler loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_pm25(
    environmental_sequence
):

    """
    environmental_sequence:
        NumPy array with shape:

        (24, 10)

    The 24 rows represent the previous
    24 hourly observations.

    The 10 columns are:

        pm25
        pm10
        no2
        so2
        o3
        co
        temperature
        relativehumidity
        wind_speed
        wind_direction
    """

    sequence = np.asarray(
        environmental_sequence,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Check shape
    # --------------------------------------------------------

    expected_shape = (
        SEQUENCE_LENGTH,
        len(FEATURES)
    )

    if sequence.shape != expected_shape:

        raise ValueError(
            f"Expected sequence shape "
            f"{expected_shape}, "
            f"but received {sequence.shape}"
        )

    # --------------------------------------------------------
    # Add batch dimension
    #
    # (24, 10)
    #      ↓
    # (1, 24, 10)
    # --------------------------------------------------------

    sequence = np.expand_dims(
        sequence,
        axis=0
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    predicted_scaled = model.predict(
        sequence,
        verbose=0
    )

    # --------------------------------------------------------
    # Convert prediction back to
    # original PM2.5 scale
    # --------------------------------------------------------

    predicted_pm25 = (
        target_scaler.inverse_transform(
            predicted_scaled
        )
    )

    predicted_pm25 = float(
        predicted_pm25[0][0]
    )

    return predicted_pm25


# ============================================================
# TEST WITH LAST TEST SEQUENCE
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("TESTING REAL LSTM PREDICTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load one real unseen test sequence.
    # --------------------------------------------------------

    X_test = np.load(
        "data/processed/X_test.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    # --------------------------------------------------------
    # Use the first test sequence.
    # --------------------------------------------------------

    sample_sequence = X_test[0]

    actual_pm25 = (
        target_scaler.inverse_transform(
            y_test[0].reshape(1, 1)
        )[0][0]
    )

    predicted_pm25 = predict_pm25(
        sample_sequence
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\nActual PM2.5:"
    )

    print(
        f"{actual_pm25:.2f} µg/m³"
    )

    print(
        "\nLSTM predicted PM2.5:"
    )

    print(
        f"{predicted_pm25:.2f} µg/m³"
    )

    print(
        "\nAbsolute error:"
    )

    print(
        f"{abs(actual_pm25 - predicted_pm25):.2f} µg/m³"
    )

    print("\n")
    print("=" * 60)
    print("PREDICTION COMPLETED")
    print("=" * 60)