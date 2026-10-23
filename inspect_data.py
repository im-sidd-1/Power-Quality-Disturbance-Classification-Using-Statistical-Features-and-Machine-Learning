
import pandas as pd

# Load the dataset
file_path = "data/Compressed PQ Data.xlsx"
df = pd.read_excel(file_path)

# Display basic information
print("\n========== DATASET INFORMATION ==========")

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\n========== DATA TYPES ==========")

print(df.dtypes)

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

print("\n========== LABEL DISTRIBUTION ==========")

print(df["label"].value_counts().sort_index())

print("\n========== UNIQUE LABELS ==========")

print(df["label"].unique())

print("\n========== DATASET SUMMARY ==========")

print(df.describe())

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== LAST 15 ROWS ==========")
print(df.tail(15).to_string())

print("\n========== INVALID LABEL ROWS ==========")

valid_labels = list(range(13))

invalid_rows = df[
    ~df["label"].isin(valid_labels)
]

print(invalid_rows.to_string())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

