
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/cleaned_pq_data.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Base model
rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

# Parameters to test
parameters = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Grid search with cross-validation
grid = GridSearchCV(
    estimator=rf,
    param_grid=parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("Starting Random Forest tuning...")

grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

print("\n========== TUNING RESULTS ==========")
print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", round(grid.best_score_ * 100, 2), "%")
print("Test Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save tuned model      
import os
import joblib

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/tuned_random_forest.pkl")

print("\nTuned model saved successfully`!")
