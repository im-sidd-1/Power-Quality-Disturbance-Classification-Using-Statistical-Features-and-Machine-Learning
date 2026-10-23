
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_pq_data.csv")

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

print("Dataset Shape:", df.shape)

print("\nClass Distribution:")
print(y.value_counts().sort_index())

# ==========================================
# 1. CLASS DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="label",
    order=sorted(df["label"].unique())
)

plt.title("Power Quality Disturbance Class Distribution")
plt.xlabel("Disturbance Label")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.savefig("data/class_distribution.png", dpi=300)
plt.show()

# ==========================================
# 2. FEATURE CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(16, 12))

correlation = X.corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("data/feature_correlation.png", dpi=300)
plt.show()

# ==========================================
# 3. FEATURE STATISTICS
# ==========================================

print("\nFeature Statistics:")
print(X.describe())

print("\nEDA Completed Successfully!")
