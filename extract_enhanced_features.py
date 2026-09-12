import pandas as pd
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

QUEUE_FILE = "output/all_fault_queues.csv"
LATENCY_FILE = "output/all_fault_latency.csv"
GROUND_TRUTH_FILE = "output/all_fault_ground_truth.csv"

OUTPUT_FILE = "output/enhanced_fault_features.csv"

WINDOW_SIZE = 100  # ms


# --------------------------------------------------
# Load data
# --------------------------------------------------

queues = pd.read_csv(QUEUE_FILE)
latency = pd.read_csv(LATENCY_FILE)
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

print("Queues:", len(queues))
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
# Helper: maximum consecutive non-zero samples
# --------------------------------------------------

def max_consecutive_nonzero(values):

    max_run = 0
    current_run = 0

    for value in values:

        if value > 0:
            current_run += 1
            max_run = max(max_run, current_run)

        else:
            current_run = 0

    return max_run


# --------------------------------------------------
# Helper: number of transitions
# --------------------------------------------------

def count_transitions(values):

    if len(values) <= 1:
        return 0

    return np.sum(
        np.diff(values) != 0
    )


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

features = []

max_time = queues["time"].max()

window_starts = np.arange(
    0,
    max_time,
    WINDOW_SIZE
)


for run_id in sorted(queues["run_id"].unique()):

    run_queues = queues[
        queues["run_id"] == run_id
    ].copy()

    run_latency = latency[
        latency["run_id"] == run_id
    ].copy()

    run_ground_truth = ground_truth[
        ground_truth["run_id"] == run_id
    ].copy()


    for window_start in window_starts:

        window_end = window_start + WINDOW_SIZE


        # ------------------------------------------
        # Queue samples
        # ------------------------------------------

        q_window = run_queues[
            (run_queues["time"] >= window_start) &
            (run_queues["time"] < window_end)
        ]


        # Skip empty windows

        if q_window.empty:
            continue


        row = {
            "run_id": run_id,
            "window_start": window_start,
            "window_end": window_end
        }


        # ------------------------------------------
        # Queue features
        # ------------------------------------------

        for queue in queue_columns:

            values = q_window[queue].values

            mean_value = np.mean(values)

            row[f"{queue}_mean"] = mean_value
            row[f"{queue}_max"] = np.max(values)
            row[f"{queue}_std"] = np.std(values)

            row[f"{queue}_nonzero"] = np.mean(
                values > 0
            )

            # P95
            row[f"{queue}_p95"] = np.percentile(
                values,
                95
            )

            # P99
            row[f"{queue}_p99"] = np.percentile(
                values,
                99
            )

            # Maximum consecutive non-zero samples
            row[f"{queue}_max_nonzero_run"] = (
                max_consecutive_nonzero(values)
            )

            # Number of queue-depth transitions
            row[f"{queue}_transitions"] = (
                count_transitions(values)
            )

            # Fraction of samples above the mean
            row[f"{queue}_above_mean"] = np.mean(
                values > mean_value
            )


        # ------------------------------------------
        # Latency features
        # ------------------------------------------

        latency_window = run_latency[
            (run_latency["finish_time"] >= window_start) &
            (run_latency["finish_time"] < window_end)
        ]


        if not latency_window.empty:

            latency_values = (
                latency_window["latency_ms"].values
            )

            row["latency_mean"] = np.mean(
                latency_values
            )

            row["latency_max"] = np.max(
                latency_values
            )

            row["latency_p95"] = np.percentile(
                latency_values,
                95
            )

            row["latency_p99"] = np.percentile(
                latency_values,
                99
            )

            row["latency_std"] = np.std(
                latency_values
            )

        else:

            row["latency_mean"] = 0
            row["latency_max"] = 0
            row["latency_p95"] = 0
            row["latency_p99"] = 0
            row["latency_std"] = 0


        # ------------------------------------------
        # Ground-truth label
        # ------------------------------------------

        label = "baseline"


        for _, fault in run_ground_truth.iterrows():

            fault_start = fault["start_time"]
            fault_end = fault["end_time"]

            overlap = (
                window_start < fault_end
                and window_end > fault_start
            )

            if overlap:

                label = fault["fault"]
                break


        row["label"] = label

        features.append(row)


# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

features_df = pd.DataFrame(features)


# --------------------------------------------------
# Save
# --------------------------------------------------

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n========================================")
print("ENHANCED FEATURE EXTRACTION COMPLETE")
print("========================================")

print("Rows:", len(features_df))
print("Columns:", len(features_df.columns))

print("\nClass distribution:")
print(
    features_df["label"].value_counts()
)

print("\nOutput:")
print(OUTPUT_FILE)