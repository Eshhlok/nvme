import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


PREDICTION_FILE = "output/xgboost_oof_predictions.csv"


# --------------------------------------------------
# Load out-of-fold predictions
# --------------------------------------------------

df = pd.read_csv(PREDICTION_FILE)

print("Rows:", len(df))


# --------------------------------------------------
# Labels
# --------------------------------------------------

labels = [
    "baseline",
    "cpu_contention",
    "network_congestion",
    "noisy_neighbour",
    "pcie_contention",
    "ssd_saturation"
]


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n=== XGBOOST CLASSIFICATION REPORT ===\n")

print(
    classification_report(
        df["label"],
        df["prediction"],
        labels=labels,
        digits=4
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    df["label"],
    df["prediction"],
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n=== XGBOOST CONFUSION MATRIX ===\n")

print(cm_df)


# --------------------------------------------------
# Error analysis
# --------------------------------------------------

errors = df[
    df["label"] != df["prediction"]
].copy()

print("\n=== TOTAL ERRORS ===")
print(len(errors))


# Actual → predicted combinations
error_pairs = (
    errors
    .groupby(["label", "prediction"])
    .size()
    .sort_values(ascending=False)
)

print("\n=== MOST COMMON ERRORS ===")

for (actual, predicted), count in error_pairs.items():

    print(
        f"{actual:20s} -> "
        f"{predicted:20s}: {count}"
    )


# --------------------------------------------------
# Save confusion matrix
# --------------------------------------------------

cm_df.to_csv(
    "output/xgboost_confusion_matrix.csv"
)

print(
    "\nSaved: output/xgboost_confusion_matrix.csv"
)