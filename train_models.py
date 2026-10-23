
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import joblib

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/cleaned_pq_data.csv")

X = df.drop(columns=["label"])
y = df["label"]

print("Dataset Shape:", df.shape)

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ==========================================
# 3. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

# ==========================================
# 4. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 5. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ==========================================
# 6. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=sorted(y.unique())
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=sorted(y.unique())
)

fig, ax = plt.subplots(figsize=(12, 10))

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation="vertical"
)

plt.title("Random Forest - Power Quality Classification")

plt.tight_layout()

plt.savefig(
    "data/random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()

# ==========================================
# 7. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")

print(importance.head(10))

# ==========================================
# 8. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

print("\nModel saved successfully!")

