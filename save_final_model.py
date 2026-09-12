import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

INPUT_FILE = "output/overlap_fault_features.csv"
MODEL_FILE = "output/final_fault_localization_model.joblib"

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# --------------------------------------------------
# Select queue features
# --------------------------------------------------

feature_cols = [
    col for col in df.columns
    if "_q_" in col
]

X = df[feature_cols]
y = df["label"]

# --------------------------------------------------
# Train final model on ALL runs
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X, y)

# --------------------------------------------------
# Save model + feature list
# --------------------------------------------------

model_package = {
    "model": model,
    "features": feature_cols
}

joblib.dump(
    model_package,
    MODEL_FILE
)

print("=== FINAL MODEL ===")
print(f"Training samples : {len(X)}")
print(f"Features         : {len(feature_cols)}")
print(f"Classes          : {sorted(y.unique())}")

print("\nModel saved to:")
print(MODEL_FILE)