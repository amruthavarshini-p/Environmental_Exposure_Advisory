import pandas as pd


FILE = "data/raw/bengaluru_environmental_2025.csv"


print("=" * 60)
print("FULL DATASET VALIDATION")
print("=" * 60)


# Load dataset
df = pd.read_csv(FILE)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True
)


print("\n1. Dataset shape:")
print(df.shape)


print("\n2. Locations:")
print(df["location"].value_counts())


print("\n3. Date range:")

print(
    "Start:",
    df["datetime"].min()
)

print(
    "End:",
    df["datetime"].max()
)


print("\n4. Missing values:")

missing = df.isnull().sum()

print(missing)


print("\n5. Missing percentage:")

missing_percent = (
    df.isnull().mean() * 100
).round(2)

print(missing_percent)


print("\n6. PM2.5 availability by location:")

pm25_by_location = (
    df.groupby("location")["pm25"]
    .agg(
        total="size",
        available="count"
    )
)

pm25_by_location["availability_%"] = (
    pm25_by_location["available"]
    / pm25_by_location["total"]
    * 100
).round(2)

print(pm25_by_location)


print("\n7. Complete rows for core pollutants:")

core_features = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co"
]

complete_core = df[
    core_features
].notna().all(axis=1)

print(
    "Rows with all core pollutants:",
    complete_core.sum()
)

print(
    "Percentage:",
    round(
        complete_core.mean() * 100,
        2
    )
)


print("\n8. Duplicate location + timestamp:")

duplicates = df.duplicated(
    subset=["location", "datetime"]
).sum()

print(duplicates)


print("\n9. Time gaps by location:")

for location in df["location"].unique():

    temp = df[
        df["location"] == location
    ].sort_values("datetime")

    gaps = (
        temp["datetime"]
        .diff()
        .value_counts()
        .head(10)
    )

    print("\n", location)
    print(gaps)


print("\n10. Basic statistics:")

print(
    df[
        [
            "pm25",
            "pm10",
            "no2",
            "so2",
            "o3",
            "co"
        ]
    ].describe()
)


print("\n")
print("=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)