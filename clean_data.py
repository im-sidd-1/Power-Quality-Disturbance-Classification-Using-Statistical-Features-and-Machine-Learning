
import pandas as pd

# Load original Excel file
file_path = "data/Compressed PQ Data.xlsx"
df = pd.read_excel(file_path)

print("Original Shape:", df.shape)

# Keep only the 750 actual samples
df = df.iloc[:750].copy()

# Remove unnecessary Excel column
df = df.drop(columns=["Unnamed: 22"], errors="ignore")

# Define feature columns
feature_columns = [
    "X2", "X7", "X12", "X15", "X16",
    "X24", "X25", "X26", "X28", "X29",
    "X33", "X35", "X37", "X39", "X40",
    "X41", "X42", "X51", "X57", "X60", "X61"
]

# Convert features to numeric values
df[feature_columns] = df[feature_columns].apply(
    pd.to_numeric, errors="coerce"
)

# Convert labels to numeric values
df["label"] = pd.to_numeric(df["label"], errors="coerce")

# Remove rows with missing values
df = df.dropna(subset=feature_columns + ["label"])

# Convert labels to integers
df["label"] = df["label"].astype(int)

# Save cleaned dataset
output_path = "data/cleaned_pq_data.csv"
df.to_csv(output_path, index=False)

print("\n========== CLEANED DATASET ==========")

print("Shape:", df.shape)

print("\nFeatures:", len(feature_columns))

print("\nUnique Labels:")
print(sorted(df["label"].unique()))

print("\nClass Distribution:")
print(df["label"].value_counts().sort_index())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nSaved to:", output_path)
