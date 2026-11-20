import os
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"

SEED = 42

EPOCHS = 50
BATCH_SIZE = 32


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

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
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nTesting:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

sequence_length = X_train.shape[1]
number_of_features = X_train.shape[2]

print("\n")
print("=" * 60)
print("GRU MODEL CONFIGURATION")
print("=" * 60)

print(
    "Sequence length:",
    sequence_length
)

print(
    "Number of input features:",
    number_of_features
)


# ============================================================
# BUILD GRU MODEL
# ============================================================

model = Sequential([

    GRU(
        64,
        return_sequences=True,
        input_shape=(
            sequence_length,
            number_of_features
        )
    ),

    Dropout(0.2),

    GRU(
        32,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

print("\n")
print("=" * 60)
print("GRU MODEL")
print("=" * 60)

model.summary()


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    f"{MODEL_DIR}/gru_best.keras",
    monitor="val_loss",
    save_best_only=True
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING GRU")
print("=" * 60)

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


# ============================================================
# LOAD BEST MODEL
# ============================================================

model = tf.keras.models.load_model(
    f"{MODEL_DIR}/gru_best.keras"
)


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("GENERATING TEST PREDICTIONS")
print("=" * 60)

y_pred_scaled = model.predict(
    X_test,
    verbose=0
).flatten()


# ============================================================
# LOAD TARGET SCALER
# ============================================================

with open(
    f"{DATA_DIR}/target_scaler.pkl",
    "rb"
) as file:

    target_scaler = pickle.load(
        file
    )


# ============================================================
# CONVERT BACK TO ORIGINAL PM2.5 SCALE
# ============================================================

y_test_original = (
    target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()
)

y_pred_original = (
    target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()
)


# ============================================================
# CALCULATE METRICS
# ============================================================

mae = mean_absolute_error(
    y_test_original,
    y_pred_original
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_original,
        y_pred_original
    )
)

r2 = r2_score(
    y_test_original,
    y_pred_original
)


# ============================================================
# TOLERANCE ACCURACY
# ============================================================

absolute_error = np.abs(
    y_test_original -
    y_pred_original
)

within_5 = (
    np.mean(
        absolute_error <= 5
    ) * 100
)

within_10 = (
    np.mean(
        absolute_error <= 10
    ) * 100
)

within_15 = (
    np.mean(
        absolute_error <= 15
    ) * 100
)


# ============================================================
# PRINT TEST RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("GRU TEST RESULTS")
print("=" * 60)

print(
    f"\nMAE  : {mae:.4f} µg/m³"
)

print(
    f"RMSE : {rmse:.4f} µg/m³"
)

print(
    f"R²   : {r2:.4f}"
)

print(
    f"\nPredictions within ±5 µg/m³  : {within_5:.2f}%"
)

print(
    f"Predictions within ±10 µg/m³ : {within_10:.2f}%"
)

print(
    f"Predictions within ±15 µg/m³ : {within_15:.2f}%"
)


# ============================================================
# TRAINING INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING INFORMATION")
print("=" * 60)

print(
    "Epochs completed:",
    len(history.history["loss"])
)

print(
    "Best validation loss:",
    min(
        history.history["val_loss"]
    )
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    f"{MODEL_DIR}/gru_final.keras"
)

print("\nFinal model saved:")
print(
    f"{MODEL_DIR}/gru_final.keras"
)

print("\nBest model saved:")
print(
    f"{MODEL_DIR}/gru_best.keras"
)

print("\n")
print("=" * 60)
print("GRU TRAINING COMPLETED")
print("=" * 60)