import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix


FEATURE_FILE = "output/overlap_fault_features.csv"
OUTPUT_FILE = "output/overlap_fault_analysis_results.csv"


df = pd.read_csv(FEATURE_FILE)

# Use all enhanced queue features
queue_features = [
    col for col in df.columns
    if "_q_" in col
]

X = df[queue_features]
y = df["label"]
groups = df["run_id"]

print("========================================")
print("50% OVERLAP FAULT-BY-FAULT ANALYSIS")
print("========================================")

print("Rows:", len(df))
print("Queue features:", len(queue_features))
print("Runs:", df["run_id"].nunique())

print("\nClass distribution:")
print(y.value_counts())


# ---------------------------------------------------------
# 5-fold run-level cross-validation
# ---------------------------------------------------------

group_kfold = GroupKFold(n_splits=5)

all_true = []
all_pred = []

for fold, (train_idx, test_idx) in enumerate(
    group_kfold.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    all_true.extend(y_test)
    all_pred.extend(predictions)


all_true = np.array(all_true)
all_pred = np.array(all_pred)


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

labels = sorted(y.unique())

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    all_true,
    all_pred,
    labels=labels,
    digits=3
)

print(report)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm_df)


# ---------------------------------------------------------
# Fault-by-fault error analysis
# ---------------------------------------------------------

print("\n========================================")
print("ERROR ANALYSIS")
print("========================================")

for actual in labels:

    total = np.sum(all_true == actual)

    correct = np.sum(
        (all_true == actual) &
        (all_pred == actual)
    )

    errors = total - correct

    recall = correct / total if total > 0 else 0

    print(f"\n{actual}")
    print("Total:", total)
    print("Correct:", correct)
    print("Errors:", errors)
    print(f"Recall: {recall:.4f}")

    if errors > 0:

        mask = (
            (all_true == actual) &
            (all_pred != actual)
        )

        error_counts = pd.Series(
            all_pred[mask]
        ).value_counts()

        print("Misclassified as:")
        print(error_counts)


# ---------------------------------------------------------
# Most common errors
# ---------------------------------------------------------

print("\n========================================")
print("MOST COMMON ERRORS")
print("========================================")

error_pairs = []

for actual, predicted in zip(all_true, all_pred):

    if actual != predicted:
        error_pairs.append(
            (actual, predicted)
        )

if error_pairs:

    error_counts = (
        pd.Series(error_pairs)
        .value_counts()
    )

    for (actual, predicted), count in error_counts.items():

        print(
            f"{actual} -> {predicted}: {count}"
        )

else:

    print("No classification errors.")


# ---------------------------------------------------------
# Save per-window predictions
# ---------------------------------------------------------

prediction_df = df.copy()

prediction_df["prediction"] = all_pred

prediction_df["correct"] = (
    prediction_df["label"] ==
    prediction_df["prediction"]
)

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("ANALYSIS COMPLETE")
print("========================================")

print("Predictions saved to:")
print(OUTPUT_FILE)