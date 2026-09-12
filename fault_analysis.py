import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import GroupKFold


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("output/all_fault_features.csv")

queue_features = [
    col for col in df.columns
    if "_q_" in col
]

X = df[queue_features]
y = df["label"]
groups = df["run_id"]

labels = sorted(y.unique())


# --------------------------------------------------
# 2. 5-fold run-level cross validation
# --------------------------------------------------

gkf = GroupKFold(n_splits=5)

all_predictions = []
all_actual = []

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups),
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

    y_pred = model.predict(X_test)

    all_actual.extend(y_test)
    all_predictions.extend(y_pred)


# --------------------------------------------------
# 3. Overall classification report
# --------------------------------------------------

print("\n")
print("=" * 70)
print("CROSS-VALIDATION CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_actual,
        all_predictions,
        labels=labels,
        zero_division=0
    )
)


# --------------------------------------------------
# 4. Combined confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_actual,
    all_predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n")
print("=" * 70)
print("COMBINED CONFUSION MATRIX")
print("=" * 70)

print(cm_df)


# --------------------------------------------------
# 5. Per-class analysis
# --------------------------------------------------

report = classification_report(
    all_actual,
    all_predictions,
    labels=labels,
    output_dict=True,
    zero_division=0
)

print("\n")
print("=" * 70)
print("FAULT-BY-FAULT ANALYSIS")
print("=" * 70)

for label in labels:

    total = cm_df.loc[label].sum()
    correct = cm_df.loc[label, label]
    incorrect = total - correct

    recall = report[label]["recall"]
    precision = report[label]["precision"]
    f1 = report[label]["f1-score"]

    print(f"\n{label}")
    print("-" * 40)

    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Incorrect     : {incorrect}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")

    if incorrect > 0:

        print("\nMisclassified as:")

        row = cm_df.loc[label].copy()
        row[label] = 0

        mistakes = row[row > 0].sort_values(
            ascending=False
        )

        for predicted_label, count in mistakes.items():
            print(
                f"  {predicted_label}: {int(count)}"
            )


# --------------------------------------------------
# 6. Most confused pairs
# --------------------------------------------------

print("\n")
print("=" * 70)
print("MOST CONFUSED FAULT PAIRS")
print("=" * 70)

pairs = []

for actual in labels:

    for predicted in labels:

        if actual == predicted:
            continue

        count = cm_df.loc[actual, predicted]

        if count > 0:
            pairs.append(
                (actual, predicted, int(count))
            )

pairs.sort(
    key=lambda x: x[2],
    reverse=True
)

for actual, predicted, count in pairs:

    print(
        f"{actual:20s} → "
        f"{predicted:20s}: {count}"
    )


# --------------------------------------------------
# 7. Save confusion matrix
# --------------------------------------------------

cm_df.to_csv(
    "output/cross_validation_confusion_matrix.csv"
)

print("\nConfusion matrix saved to:")
print("output/cross_validation_confusion_matrix.csv")