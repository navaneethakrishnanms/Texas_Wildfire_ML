import pandas as pd

# Load the original FIRMS dataset
df = pd.read_csv("fire_archive_M-C61_760762.csv")

# Convert acq_date to datetime
df["acq_date"] = pd.to_datetime(df["acq_date"])

# Filter only 2024 records
df_2024 = df[df["acq_date"].dt.year == 2024]

# Save the filtered dataset
df_2024.to_csv("data/raw/firms/fire_archive_2024.csv", index=False)

print("Original rows:", len(df))
print("2024 rows:", len(df_2024))
print("Saved as data/raw/firms/fire_archive_2024.csv")