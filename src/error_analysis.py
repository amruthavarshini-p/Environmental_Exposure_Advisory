import pandas as pd
import numpy as np

FILE = "data/processed/lstm_test_predictions.csv"

df = pd.read_csv(FILE)

df["absolute_error"] = np.abs(
    df["actual_pm25_ug_m3"] -
    df["predicted_pm25_ug_m3"]
)

df = df.sort_values(
    "absolute_error",
    ascending=False
)

print("=" * 60)
print("LSTM ERROR ANALYSIS")
print("=" * 60)

print("\nTop 20 largest prediction errors:")

print(
    df.head(20).to_string(index=False)
)

print("\n")
print("=" * 60)
print("ERROR SUMMARY")
print("=" * 60)

print(
    f"\nMean error: "
    f"{df['absolute_error'].mean():.4f} µg/m³"
)

print(
    f"\nMedian error: "
    f"{df['absolute_error'].median():.4f} µg/m³"
)

print(
    f"\n90th percentile error: "
    f"{df['absolute_error'].quantile(0.90):.4f} µg/m³"
)

print(
    f"\n95th percentile error: "
    f"{df['absolute_error'].quantile(0.95):.4f} µg/m³"
)

print(
    f"\nMaximum error: "
    f"{df['absolute_error'].max():.4f} µg/m³"
)