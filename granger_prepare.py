import pandas as pd
from pathlib import Path


QUEUE_PATH = "output/all_fault_queues.csv"
GROUND_TRUTH_PATH = "output/all_fault_ground_truth.csv"

OUTPUT_PATH = "output/granger_windows.csv"

PRE_FAULT_MS = 200
POST_FAULT_MS = 200

QUEUE_COLUMNS = [
    "host_cpu_q",
    "host_pcie_q",
    "host_nic_q",
    "network_q",
    "target_nic_q",
    "target_pcie_q",
    "ssd_q",
]


# ============================================================
# LOAD
# ============================================================

queues = pd.read_csv(QUEUE_PATH)
faults = pd.read_csv(GROUND_TRUTH_PATH)

queues = queues.sort_values(
    ["run_id", "time"]
).reset_index(drop=True)

faults = faults.sort_values(
    ["run_id", "start_time"]
).reset_index(drop=True)


print(f"Telemetry rows : {len(queues)}")
print(f"Fault intervals: {len(faults)}")


# ============================================================
# EXTRACT WINDOWS
# ============================================================

windows = []

for _, fault in faults.iterrows():

    run_id = int(fault["run_id"])

    fault_start = float(
        fault["start_time"]
    )

    fault_end = float(
        fault["end_time"]
    )

    fault_name = fault["fault"]

    window_start = (
        fault_start -
        PRE_FAULT_MS
    )

    window_end = (
        fault_end +
        POST_FAULT_MS
    )

    mask = (
        (queues["run_id"] == run_id)
        &
        (queues["time"] >= window_start)
        &
        (queues["time"] <= window_end)
    )

    segment = queues.loc[
        mask,
        ["run_id", "time"] + QUEUE_COLUMNS
    ].copy()

    if len(segment) == 0:
        continue

    segment["fault"] = fault_name
    segment["fault_start"] = fault_start
    segment["fault_end"] = fault_end

    # Position relative to fault onset.
    segment["relative_time"] = (
        segment["time"] -
        fault_start
    )

    windows.append(segment)


# ============================================================
# COMBINE
# ============================================================

result = pd.concat(
    windows,
    ignore_index=True
)

result = result.sort_values(
    ["run_id", "fault_start", "time"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# CHECK
# ============================================================

print("\n" + "=" * 70)
print("GRANGER WINDOW DATASET")
print("=" * 70)

print(
    f"Rows          : {len(result)}"
)

print(
    f"Fault windows : "
    f"{result[['run_id', 'fault_start']].drop_duplicates().shape[0]}"
)

print(
    f"Runs          : "
    f"{result['run_id'].nunique()}"
)

print("\nSamples per fault window:")

print(
    result
    .groupby(
        ["run_id", "fault"]
    )
    .size()
    .describe()
)

print("\nFault distribution:")

print(
    result["fault"].value_counts()
)

print("\nSaved:")
print(OUTPUT_PATH)