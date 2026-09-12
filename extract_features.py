import pandas as pd
import numpy as np
import os

os.makedirs("output", exist_ok=True)

QUEUE_FILE = "output/all_fault_queues.csv"
LATENCY_FILE = "output/all_fault_latency.csv"
GROUND_TRUTH_FILE = "output/all_fault_ground_truth.csv"

WINDOW_MS = 100

# --------------------------------------------------
# Load data
# --------------------------------------------------

queues = pd.read_csv(QUEUE_FILE)
latency = pd.read_csv(LATENCY_FILE)
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

print("Loaded data:")
print("Queue records:", len(queues))
print("Latency records:", len(latency))
print("Fault intervals:", len(ground_truth))


# --------------------------------------------------
# Queue columns
# --------------------------------------------------

queue_columns = [
    "host_cpu_q",
    "host_pcie_q",
    "host_nic_q",
    "network_q",
    "target_nic_q",
    "target_pcie_q",
    "ssd_q"
]


# --------------------------------------------------
# Determine label for a window
# --------------------------------------------------

def get_label(run_id, start_time, end_time):

    faults = ground_truth[
        (ground_truth["run_id"] == run_id)
        &
        (ground_truth["start_time"] < end_time)
        &
        (ground_truth["end_time"] > start_time)
    ]

    if len(faults) == 0:
        return "baseline"

    # A window should normally overlap only one fault.
    return faults.iloc[0]["fault"]


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

feature_rows = []

run_ids = sorted(queues["run_id"].unique())

for run_id in run_ids:

    run_queues = queues[
        queues["run_id"] == run_id
    ].copy()

    run_latency = latency[
        latency["run_id"] == run_id
    ].copy()

    max_time = max(
        run_queues["time"].max(),
        run_latency["finish_time"].max()
    )

    window_start = 0

    while window_start < max_time:

        window_end = window_start + WINDOW_MS

        # ------------------------------
        # Queue telemetry
        # ------------------------------

        q_window = run_queues[
            (run_queues["time"] >= window_start)
            &
            (run_queues["time"] < window_end)
        ]

        # ------------------------------
        # Latency telemetry
        # ------------------------------

        lat_window = run_latency[
            (run_latency["finish_time"] >= window_start)
            &
            (run_latency["finish_time"] < window_end)
        ]

        features = {
            "run_id": run_id,
            "window_start": window_start,
            "window_end": window_end
        }

        # ------------------------------
        # Queue features
        # ------------------------------

        for column in queue_columns:

            values = q_window[column]

            if len(values) == 0:
                features[f"{column}_mean"] = 0
                features[f"{column}_max"] = 0
                features[f"{column}_std"] = 0
                features[f"{column}_nonzero"] = 0

            else:
                features[f"{column}_mean"] = values.mean()
                features[f"{column}_max"] = values.max()
                features[f"{column}_std"] = values.std()
                features[f"{column}_nonzero"] = (
                    (values > 0).mean()
                )

        # ------------------------------
        # Latency features
        # ------------------------------

        if len(lat_window) == 0:

            features["latency_mean"] = 0
            features["latency_max"] = 0
            features["latency_p95"] = 0
            features["latency_std"] = 0

        else:

            lat_values = lat_window["latency_ms"]

            features["latency_mean"] = lat_values.mean()
            features["latency_max"] = lat_values.max()
            features["latency_p95"] = lat_values.quantile(0.95)
            features["latency_std"] = lat_values.std()

        # ------------------------------
        # Ground-truth label
        # ------------------------------

        features["label"] = get_label(
            run_id,
            window_start,
            window_end
        )

        feature_rows.append(features)

        window_start += WINDOW_MS


# --------------------------------------------------
# Create dataframe
# --------------------------------------------------

features_df = pd.DataFrame(feature_rows)

# Replace NaN standard deviations
features_df = features_df.fillna(0)


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_FILE = "output/all_fault_features.csv"

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("=" * 50)
print("MULTI-RUN FEATURE DATASET")
print("=" * 50)

print("Runs:", features_df["run_id"].nunique())
print("Windows:", len(features_df))

print()
print("Class distribution:")
print(features_df["label"].value_counts())

print()
print("Windows per run:")
print(features_df.groupby("run_id").size().describe())

print()
print("Saved:")
print(OUTPUT_FILE)