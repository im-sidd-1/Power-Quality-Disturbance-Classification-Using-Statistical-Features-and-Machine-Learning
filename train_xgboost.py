
import pandas as pd
import os
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/cleaned_pq_data.csv")

X = df.drop("label", axis=1)
y = df["label"].astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Initialize XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=13,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Train
print("Training XGBoost...")
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("\n========== XGBOOST PERFORMANCE ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_model.pkl")

print("\nXGBoost model saved successfully!")

# Save confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig("data/xgboost_confusion_matrix.png")
plt.close()

print("Confusion matrix saved successfully!")
