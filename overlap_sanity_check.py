import pandas as pd

GROUND_TRUTH_FILE = "output/all_fault_ground_truth.csv"
FEATURE_FILE = "output/overlap_fault_features.csv"

WINDOW_SIZE = 100
MIN_OVERLAP = 0.50


# --------------------------------------------------
# Load data
# --------------------------------------------------

features = pd.read_csv(FEATURE_FILE)
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

print("=== DATASET SANITY CHECK ===")
print(f"Feature rows      : {len(features)}")
print(f"Ground-truth rows : {len(ground_truth)}")
print(f"Runs              : {features['run_id'].nunique()}")


# --------------------------------------------------
# Recalculate expected label for every window
# --------------------------------------------------

errors = []

for _, row in features.iterrows():

    run_id = row["run_id"]
    window_start = row["window_start"]
    window_end = row["window_end"]

    run_faults = ground_truth[
        ground_truth["run_id"] == run_id
    ]

    expected_label = "baseline"
    max_overlap = 0.0
    matching_fault = None

    for _, fault in run_faults.iterrows():

        fault_start = fault["start_time"]
        fault_end = fault["end_time"]

        overlap_start = max(window_start, fault_start)
        overlap_end = min(window_end, fault_end)

        overlap_duration = max(
            0,
            overlap_end - overlap_start
        )

        overlap_fraction = overlap_duration / WINDOW_SIZE

        if overlap_fraction > max_overlap:
            max_overlap = overlap_fraction
            matching_fault = fault["fault"]

        if overlap_fraction >= MIN_OVERLAP:
            expected_label = fault["fault"]

    actual_label = row["label"]

    if actual_label != expected_label:
        errors.append({
            "run_id": run_id,
            "window_start": window_start,
            "window_end": window_end,
            "actual": actual_label,
            "expected": expected_label,
            "max_overlap": max_overlap,
            "matching_fault": matching_fault
        })


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n=== LABEL CONSISTENCY ===")

if len(errors) == 0:
    print("✓ All labels are correct.")
    print("✓ Every window follows the 50% overlap rule.")
else:
    print(f"✗ Found {len(errors)} label mismatches.")

    error_df = pd.DataFrame(errors)

    print("\nFirst mismatches:")
    print(error_df.head(20).to_string(index=False))


# --------------------------------------------------
# Class distribution
# --------------------------------------------------

print("\n=== CLASS DISTRIBUTION ===")

print(
    features["label"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# Check fault windows specifically
# --------------------------------------------------

fault_labels = [
    "cpu_contention",
    "pcie_contention",
    "ssd_saturation",
    "noisy_neighbour",
    "network_congestion"
]

print("\n=== FAULT WINDOW OVERLAP CHECK ===")

fault_windows = features[
    features["label"].isin(fault_labels)
]

overlap_values = []

for _, row in fault_windows.iterrows():

    run_id = row["run_id"]
    window_start = row["window_start"]
    window_end = row["window_end"]

    run_faults = ground_truth[
        ground_truth["run_id"] == run_id
    ]

    max_overlap = 0.0

    for _, fault in run_faults.iterrows():

        overlap_start = max(
            window_start,
            fault["start_time"]
        )

        overlap_end = min(
            window_end,
            fault["end_time"]
        )

        overlap_duration = max(
            0,
            overlap_end - overlap_start
        )

        overlap_fraction = (
            overlap_duration / WINDOW_SIZE
        )

        max_overlap = max(
            max_overlap,
            overlap_fraction
        )

    overlap_values.append(max_overlap)


overlap_values = pd.Series(overlap_values)

print(f"Minimum overlap : {overlap_values.min():.2f}")
print(f"Maximum overlap : {overlap_values.max():.2f}")
print(f"Mean overlap    : {overlap_values.mean():.2f}")

below_threshold = (
    overlap_values < MIN_OVERLAP
).sum()

print(
    f"Fault windows below 50% : {below_threshold}"
)


# --------------------------------------------------
# Boundary windows
# --------------------------------------------------

print("\n=== BOUNDARY CHECK ===")

boundary_windows = []

for _, row in features.iterrows():

    run_id = row["run_id"]
    start = row["window_start"]
    end = row["window_end"]

    run_faults = ground_truth[
        ground_truth["run_id"] == run_id
    ]

    for _, fault in run_faults.iterrows():

        overlap_start = max(start, fault["start_time"])
        overlap_end = min(end, fault["end_time"])

        overlap = max(
            0,
            overlap_end - overlap_start
        )

        fraction = overlap / WINDOW_SIZE

        # Show windows close to the decision boundary
        if 0.0 < fraction < 1.0:
            boundary_windows.append({
                "run_id": run_id,
                "window_start": start,
                "window_end": end,
                "fault": fault["fault"],
                "overlap_fraction": fraction,
                "label": row["label"]
            })

boundary_df = pd.DataFrame(boundary_windows)

if len(boundary_df) > 0:
    print(
        boundary_df
        .sort_values("overlap_fraction")
        .head(20)
        .to_string(index=False)
    )
else:
    print("No boundary windows found.")


print("\n=== CHECK COMPLETE ===")