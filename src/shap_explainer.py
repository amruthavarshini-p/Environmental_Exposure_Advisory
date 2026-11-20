import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt


# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "models/lstm_best.keras"
TEST_DATA_PATH = "data/processed/X_test.npy"

OUTPUT_DIR = "outputs/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# FEATURE NAMES
# ==========================================================

FEATURE_NAMES = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
    "CO",
    "Temperature",
    "Relative Humidity",
    "Wind Speed",
    "Wind Direction"
]


# ==========================================================
# LOAD MODEL
# ==========================================================

print("\n==========================================")
print(" SHAP EXPLAINABILITY")
print("==========================================")

print("\nLoading LSTM model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("LSTM model loaded successfully.")


# ==========================================================
# LOAD TEST DATA
# ==========================================================

print("\nLoading test data...")

X_test = np.load(
    TEST_DATA_PATH
)

print(
    f"Test data shape: {X_test.shape}"
)


# ==========================================================
# SELECT BACKGROUND DATA
# ==========================================================

# Use a small representative sample from
# the test data as the SHAP background.

background_size = min(
    20,
    len(X_test)
)

background = X_test[
    :background_size
]

print(
    f"SHAP background shape: "
    f"{background.shape}"
)


# ==========================================================
# SELECT SAMPLE TO EXPLAIN
# ==========================================================

sample_index = 0

sample = X_test[
    sample_index:sample_index + 1
]

print(
    f"Sample being explained: "
    f"{sample.shape}"
)


# ==========================================================
# CREATE SHAP EXPLAINER
# ==========================================================

print("\nCreating SHAP Gradient Explainer...")

explainer = shap.GradientExplainer(
    model,
    background
)

print("SHAP explainer created successfully.")


# ==========================================================
# CALCULATE SHAP VALUES
# ==========================================================

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(
    sample
)

print("SHAP values calculated.")


# ==========================================================
# HANDLE SHAP OUTPUT FORMAT
# ==========================================================

# Depending on the SHAP version, the output can be:
#
#   ndarray
#
# or
#
#   list containing an ndarray
#
# We normalize it here.

if isinstance(shap_values, list):

    shap_values = shap_values[0]

shap_values = np.asarray(
    shap_values
)

print(
    f"Raw SHAP shape: "
    f"{shap_values.shape}"
)


# ==========================================================
# REMOVE OUTPUT DIMENSION IF PRESENT
# ==========================================================

# Expected model input:
#
# (1, 24, 10)
#
# Some SHAP versions return:
#
# (1, 24, 10, 1)
#
# Convert it to:
#
# (1, 24, 10)

if shap_values.ndim == 4:

    if shap_values.shape[-1] == 1:

        shap_values = np.squeeze(
            shap_values,
            axis=-1
        )


# ==========================================================
# CHECK SHAPE
# ==========================================================

if shap_values.ndim != 3:

    raise ValueError(
        f"Unexpected SHAP shape: "
        f"{shap_values.shape}"
    )


# ==========================================================
# AGGREGATE SHAP VALUES ACROSS 24 HOURS
# ==========================================================

# We want feature-level importance.
#
# For each feature:
#
# 24 hourly SHAP values
#          ↓
# mean absolute contribution
#
# This gives one importance value
# for each environmental feature.

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=(0, 1)
)


# ==========================================================
# CREATE FEATURE IMPORTANCE TABLE
# ==========================================================

importance_df = pd.DataFrame(
    {
        "Feature": FEATURE_NAMES,
        "Mean_Absolute_SHAP": mean_abs_shap
    }
)

importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

importance_df = importance_df.reset_index(
    drop=True
)


# ==========================================================
# DISPLAY FEATURE IMPORTANCE
# ==========================================================

print("\n==========================================")
print(" SHAP FEATURE IMPORTANCE")
print("==========================================")

print(
    importance_df.to_string(
        index=False
    )
)


# ==========================================================
# SAVE FEATURE IMPORTANCE
# ==========================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.csv"
)

importance_df.to_csv(
    csv_path,
    index=False
)

print(
    f"\nFeature importance saved to:"
    f"\n{csv_path}"
)


# ==========================================================
# CREATE BAR CHART
# ==========================================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    importance_df["Feature"],
    importance_df["Mean_Absolute_SHAP"]
)

plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "Environmental Feature"
)

plt.title(
    "LSTM Feature Importance Using SHAP"
)

plt.gca().invert_yaxis()

plt.tight_layout()


# ==========================================================
# SAVE CHART
# ==========================================================

plot_path = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"SHAP plot saved to:"
    f"\n{plot_path}"
)


# ==========================================================
# SAVE RAW SHAP VALUES
# ==========================================================

raw_shap_path = os.path.join(
    OUTPUT_DIR,
    "shap_values.npy"
)

np.save(
    raw_shap_path,
    shap_values
)

print(
    f"Raw SHAP values saved to:"
    f"\n{raw_shap_path}"
)


# ==========================================================
# COMPLETION
# ==========================================================

print("\n==========================================")
print(" SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("==========================================")