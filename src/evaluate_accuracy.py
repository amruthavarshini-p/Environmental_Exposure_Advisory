import pandas as pd
import numpy as np


# ============================================================
# LOAD LSTM TEST PREDICTIONS
# ============================================================

FILE = "data/processed/lstm_test_predictions.csv"

df = pd.read_csv(FILE)

actual = df["actual_pm25_ug_m3"].values
predicted = df["predicted_pm25_ug_m3"].values


# ============================================================
# CALCULATE TOLERANCE-BASED ACCURACY
# ============================================================

def accuracy_within_tolerance(
    actual,
    predicted,
    tolerance
):

    error = np.abs(
        actual - predicted
    )

    accuracy = (
        np.mean(
            error <= tolerance
        ) * 100
    )

    return accuracy


accuracy_5 = accuracy_within_tolerance(
    actual,
    predicted,
    5
)

accuracy_10 = accuracy_within_tolerance(
    actual,
    predicted,
    10
)

accuracy_15 = accuracy_within_tolerance(
    actual,
    predicted,
    15
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 60)
print("LSTM PREDICTION ACCURACY ANALYSIS")
print("=" * 60)

print(
    f"\nTotal test predictions: {len(df)}"
)

print(
    f"\nPredictions within ±5 µg/m³:"
)

print(
    f"{accuracy_5:.2f}%"
)

print(
    f"\nPredictions within ±10 µg/m³:"
)

print(
    f"{accuracy_10:.2f}%"
)

print(
    f"\nPredictions within ±15 µg/m³:"
)

print(
    f"{accuracy_15:.2f}%"
)


# ============================================================
# ERROR STATISTICS
# ============================================================

absolute_error = np.abs(
    actual - predicted
)

print("\n")
print("=" * 60)
print("ERROR DISTRIBUTION")
print("=" * 60)

print(
    f"\nMean absolute error:"
    f" {absolute_error.mean():.4f} µg/m³"
)

print(
    f"\nMedian absolute error:"
    f" {np.median(absolute_error):.4f} µg/m³"
)

print(
    f"\nMaximum absolute error:"
    f" {absolute_error.max():.4f} µg/m³"
)


print("\n")
print("=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)