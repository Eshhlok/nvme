import pandas as pd
import joblib

from xgboost import XGBClassifier


INPUT_PATH = "output/overlap_fault_features.csv"
OUTPUT_PATH = "output/final_xgboost_fault_localization_model.joblib"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=== FINAL XGBOOST MODEL ===")
print(f"Rows: {len(df)}")


# ============================================================
# FEATURES
# ============================================================

feature_cols = [
    col
    for col in df.columns
    if "_q_" in col
]

X = df[feature_cols]
y_text = df["label"]


# ============================================================
# ENCODE LABELS
# ============================================================

class_names = sorted(
    y_text.unique()
)

label_to_id = {
    label: idx
    for idx, label in enumerate(class_names)
}

y = y_text.map(
    label_to_id
).values


print("\nClasses:")

for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")


print(
    f"\nFeatures: {len(feature_cols)}"
)


# ============================================================
# FINAL MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(class_names),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN ON ALL DATA
# ============================================================

print("\nTraining final model...")

model.fit(
    X,
    y
)


# ============================================================
# SAVE
# ============================================================

bundle = {
    "model": model,
    "features": feature_cols,
    "class_names": class_names,
    "label_to_id": label_to_id
}

joblib.dump(
    bundle,
    OUTPUT_PATH
)

print(
    f"\nSaved final model:"
    f"\n{OUTPUT_PATH}"
)