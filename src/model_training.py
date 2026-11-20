import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"

SEED = 42

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)

    X_train = np.load(
        f"{DATA_DIR}/X_train.npy"
    )

    y_train = np.load(
        f"{DATA_DIR}/y_train.npy"
    )

    X_val = np.load(
        f"{DATA_DIR}/X_val.npy"
    )

    y_val = np.load(
        f"{DATA_DIR}/y_val.npy"
    )

    X_test = np.load(
        f"{DATA_DIR}/X_test.npy"
    )

    y_test = np.load(
        f"{DATA_DIR}/y_test.npy"
    )

    print("\nTraining:")
    print("X:", X_train.shape)
    print("y:", y_train.shape)

    print("\nValidation:")
    print("X:", X_val.shape)
    print("y:", y_val.shape)

    print("\nTesting:")
    print("X:", X_test.shape)
    print("y:", y_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# LOAD SCALERS
# ============================================================

def load_scalers():

    print("\n")
    print("=" * 60)
    print("LOADING SCALERS")
    print("=" * 60)

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

    print("\nScalers loaded successfully.")

    return (
        feature_scaler,
        target_scaler
    )


# ============================================================
# CONVERT PM2.5 BACK TO ORIGINAL UNITS
# ============================================================

def inverse_pm25_scaling(
    values,
    feature_scaler,
    target_scaler
):

    values = np.asarray(
        values
    ).reshape(-1, 1)

    # The current preprocessing first scaled PM2.5
    # as part of FEATURES and then fitted the target scaler
    # on that scaled PM2.5.
    #
    # Therefore we reverse the transformations in the
    # opposite order.

    values = target_scaler.inverse_transform(
        values
    )

    # PM2.5 is feature index 0.

    original_min = feature_scaler.data_min_[0]
    original_max = feature_scaler.data_max_[0]

    values = (
        values *
        (original_max - original_min)
        +
        original_min
    )

    return values.flatten()


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def build_lstm_model(
    input_shape
):

    print("\n")
    print("=" * 60)
    print("BUILDING LSTM MODEL")
    print("=" * 60)

    model = Sequential([

        LSTM(
            64,
            return_sequences=True,
            input_shape=input_shape
        ),

        Dropout(0.2),

        LSTM(
            32
        ),

        Dropout(0.2),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1
        )
    ])

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    model.summary()

    return model


# ============================================================
# TRAIN
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\n")
    print("=" * 60)
    print("TRAINING LSTM MODEL")
    print("=" * 60)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = (
        f"{MODEL_DIR}/lstm_pm25_best.keras"
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        verbose=1
    )

    print("\nBest model saved:")
    print(model_path)

    return history


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    feature_scaler,
    target_scaler
):

    print("\n")
    print("=" * 60)
    print("EVALUATING LSTM MODEL")
    print("=" * 60)

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    ).flatten()

    # Convert both actual and predicted values
    # back to original PM2.5 units.

    actual_pm25 = inverse_pm25_scaling(
        y_test,
        feature_scaler,
        target_scaler
    )

    predicted_pm25 = inverse_pm25_scaling(
        predictions_scaled,
        feature_scaler,
        target_scaler
    )

    # --------------------------------------------------------
    # Metrics in original µg/m³
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual_pm25,
        predicted_pm25
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_pm25,
            predicted_pm25
        )
    )

    r2 = r2_score(
        actual_pm25,
        predicted_pm25
    )

    print("\nFINAL TEST METRICS")
    print("-" * 40)

    print(
        f"MAE  : {mae:.4f} µg/m³"
    )

    print(
        f"RMSE : {rmse:.4f} µg/m³"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_file = (
        f"{DATA_DIR}/"
        "lstm_test_predictions.csv"
    )

    predictions_df = pd.DataFrame({

        "actual_pm25_ug_m3":
            actual_pm25,

        "predicted_pm25_ug_m3":
            predicted_pm25

    })

    predictions_df.to_csv(
        prediction_file,
        index=False
    )

    print("\nPredictions saved:")
    print(prediction_file)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(history):

    history_df = pd.DataFrame(
        history.history
    )

    history_file = (
        f"{DATA_DIR}/"
        "lstm_training_history.csv"
    )

    history_df.to_csv(
        history_file,
        index=False
    )

    print("\nTraining history saved:")
    print(history_file)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    # --------------------------------------------------------
    # Load scalers
    # --------------------------------------------------------

    (
        feature_scaler,
        target_scaler
    ) = load_scalers()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_lstm_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    save_history(
        history
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        feature_scaler,
        target_scaler
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("LSTM TRAINING COMPLETED")
    print("=" * 60)

    print("\nFinal test metrics:")

    print(
        f"MAE  : {metrics['MAE']:.4f} µg/m³"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f} µg/m³"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()