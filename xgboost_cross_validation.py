import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score


INPUT_FILE = "output/overlap_fault_features.csv"

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# Same feature selection used for Random Forest
feature_cols = [
    col for col in df.columns
    if "_q_" in col
]

X = df[feature_cols]
y = df["label"]
groups = df["run_id"]

print("=== XGBOOST DATASET ===")
print(f"Rows     : {len(df)}")
print(f"Features : {len(feature_cols)}")
print(f"Runs     : {df['run_id'].nunique()}")
print(f"Classes  : {sorted(y.unique())}")


# --------------------------------------------------
# Encode labels
# --------------------------------------------------

class_names = sorted(y.unique())

label_to_id = {
    label: i
    for i, label in enumerate(class_names)
}

id_to_label = {
    i: label
    for label, i in label_to_id.items()
}

y_encoded = y.map(label_to_id)


# --------------------------------------------------
# 5-fold run-level cross-validation
# --------------------------------------------------

cv = GroupKFold(n_splits=5)

accuracies = []
macro_f1_scores = []

oof_predictions = np.empty(len(df), dtype=int)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y_encoded, groups),
    start=1
):

    print(f"\n=== Fold {fold} ===")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(class_names),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X.iloc[train_idx],
        y_encoded.iloc[train_idx]
    )

    predictions = model.predict(
        X.iloc[test_idx]
    )

    oof_predictions[test_idx] = predictions

    accuracy = accuracy_score(
        y_encoded.iloc[test_idx],
        predictions
    )

    macro_f1 = f1_score(
        y_encoded.iloc[test_idx],
        predictions,
        average="macro"
    )

    accuracies.append(accuracy)
    macro_f1_scores.append(macro_f1)

    train_runs = sorted(
        df.iloc[train_idx]["run_id"].unique()
    )

    test_runs = sorted(
        df.iloc[test_idx]["run_id"].unique()
    )

    print(
        f"Train runs : {train_runs}"
    )

    print(
        f"Test runs  : {test_runs}"
    )

    print(
        f"Accuracy   : {accuracy:.4f}"
    )

    print(
        f"Macro F1   : {macro_f1:.4f}"
    )


# --------------------------------------------------
# Overall results
# --------------------------------------------------

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

mean_f1 = np.mean(macro_f1_scores)
std_f1 = np.std(macro_f1_scores)

print("\n" + "=" * 50)
print("XGBOOST CROSS-VALIDATION RESULTS")
print("=" * 50)

for i in range(5):
    print(
        f"Fold {i + 1}: "
        f"Accuracy={accuracies[i]:.4f}, "
        f"Macro F1={macro_f1_scores[i]:.4f}"
    )

print("\nMean Accuracy :", f"{mean_accuracy:.4f}")
print("Std Accuracy  :", f"{std_accuracy:.4f}")

print("\nMean Macro F1 :", f"{mean_f1:.4f}")
print("Std Macro F1  :", f"{std_f1:.4f}")


# --------------------------------------------------
# Save fold results
# --------------------------------------------------

results = pd.DataFrame({
    "fold": range(1, 6),
    "accuracy": accuracies,
    "macro_f1": macro_f1_scores
})

results.to_csv(
    "output/xgboost_cv_results.csv",
    index=False
)

# --------------------------------------------------
# Save out-of-fold predictions
# --------------------------------------------------

prediction_df = df[
    ["run_id", "window_start", "window_end", "label"]
].copy()

prediction_df["prediction"] = [
    id_to_label[p]
    for p in oof_predictions
]

prediction_df.to_csv(
    "output/xgboost_oof_predictions.csv",
    index=False
)

print("\nSaved:")
print("output/xgboost_cv_results.csv")
print("output/xgboost_oof_predictions.csv")