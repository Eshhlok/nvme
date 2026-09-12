import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold


FEATURE_FILE = "output/enhanced_fault_features.csv"
GROUND_TRUTH_FILE = "output/all_fault_ground_truth.csv"

df = pd.read_csv(FEATURE_FILE)
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

queue_features = [
    col for col in df.columns
    if "_q_" in col
]

X = df[queue_features]
y = df["label"]
groups = df["run_id"]

group_kfold = GroupKFold(n_splits=5)

all_records = []

for fold, (train_idx, test_idx) in enumerate(
    group_kfold.split(X, y, groups), start=1
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

    test_df = df.iloc[test_idx].copy()
    test_df["prediction"] = predictions

    all_records.append(test_df)


results = pd.concat(all_records, ignore_index=True)

# Only analyze actual fault windows
fault_results = results[
    results["label"] != "baseline"
].copy()


# ---------------------------------------------------------
# Determine where each window lies inside its fault interval
# ---------------------------------------------------------

def get_fault_position(row):

    run_faults = ground_truth[
        ground_truth["run_id"] == row["run_id"]
    ]

    for _, fault in run_faults.iterrows():

        fault_start = fault["start_time"]
        fault_end = fault["end_time"]

        overlap = (
            row["window_start"] < fault_end
            and row["window_end"] > fault_start
        )

        if overlap and fault["label"] if False else False:
            pass

        if overlap and row["label"] == fault["fault"]:

            fault_duration = fault_end - fault_start

            # Position of the window midpoint
            midpoint = (
                row["window_start"] +
                row["window_end"]
            ) / 2

            position = (
                midpoint - fault_start
            ) / fault_duration

            return max(0, min(1, position))

    return np.nan


fault_results["fault_position"] = fault_results.apply(
    get_fault_position,
    axis=1
)


# ---------------------------------------------------------
# Create position bins
# ---------------------------------------------------------

bins = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]

labels = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%"
]

fault_results["position_bin"] = pd.cut(
    fault_results["fault_position"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# ---------------------------------------------------------
# Overall accuracy by fault position
# ---------------------------------------------------------

fault_results["correct"] = (
    fault_results["label"] ==
    fault_results["prediction"]
)

print("========================================")
print("FAULT POSITION ANALYSIS")
print("========================================")

print("\nOverall accuracy by position:")
print(
    fault_results
    .groupby("position_bin", observed=False)["correct"]
    .agg(["count", "mean"])
)


# ---------------------------------------------------------
# Accuracy by fault AND position
# ---------------------------------------------------------

print("\n========================================")
print("ACCURACY BY FAULT AND POSITION")
print("========================================")

position_table = (
    fault_results
    .groupby(
        ["label", "position_bin"],
        observed=False
    )["correct"]
    .agg(["count", "mean"])
)

print(position_table)


# ---------------------------------------------------------
# Error rate at beginning / middle / end
# ---------------------------------------------------------

fault_results["region"] = pd.cut(
    fault_results["fault_position"],
    bins=[0, 0.2, 0.8, 1.0],
    labels=[
        "fault_start_0_20%",
        "fault_middle_20_80%",
        "fault_end_80_100%"
    ],
    include_lowest=True
)

print("\n========================================")
print("START vs MIDDLE vs END")
print("========================================")

print(
    fault_results
    .groupby("region", observed=False)["correct"]
    .agg(["count", "mean"])
)


# ---------------------------------------------------------
# Fault-specific start/middle/end
# ---------------------------------------------------------

print("\n========================================")
print("FAULT-SPECIFIC START/MIDDLE/END")
print("========================================")

print(
    fault_results
    .groupby(
        ["label", "region"],
        observed=False
    )["correct"]
    .agg(["count", "mean"])
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

fault_results.to_csv(
    "output/fault_position_results.csv",
    index=False
)

print("\nResults saved to:")
print("output/fault_position_results.csv")