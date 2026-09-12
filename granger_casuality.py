import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import grangercausalitytests


# ============================================================
# CONFIG
# ============================================================

QUEUE_PATH = "output/all_fault_queues.csv"
GROUND_TRUTH_PATH = "output/all_fault_ground_truth.csv"

OUTPUT_PATH = "output/granger_results.csv"

# Test delays of 1, 5, 10 and 20 ms.
MAX_LAG = 20
TEST_LAGS = [1, 5, 10, 20]

MIN_SAMPLES = 100

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
# LOAD DATA
# ============================================================

print("Loading telemetry...")

queues = pd.read_csv(
    QUEUE_PATH
)

ground_truth = pd.read_csv(
    GROUND_TRUTH_PATH
)

print(f"Queue records: {len(queues)}")
print(f"Ground-truth intervals: {len(ground_truth)}")
print(f"Runs: {queues['run_id'].nunique()}")


# ============================================================
# PREPARE
# ============================================================

queues = queues.sort_values(
    ["run_id", "time"]
).reset_index(drop=True)

ground_truth = ground_truth.sort_values(
    ["run_id", "start_time"]
).reset_index(drop=True)


# ============================================================
# EXTRACT FAULT SERIES
# ============================================================

def get_fault_series(
    run_id,
    fault_start,
    fault_end
):

    # Include the fault interval itself.
    mask = (
        (queues["run_id"] == run_id)
        &
        (queues["time"] >= fault_start)
        &
        (queues["time"] <= fault_end)
    )

    data = queues.loc[
        mask,
        QUEUE_COLUMNS
    ].copy()

    return data


# ============================================================
# GRANGER TEST
# ============================================================

def run_granger_test(
    source,
    target,
    data,
    lag
):

    pair = data[
        [target, source]
    ].copy()

    # Remove invalid values.
    pair = pair.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(pair) < MIN_SAMPLES:
        return np.nan

    # Constant signals cannot provide useful
    # Granger information.
    if pair[source].std() < 1e-12:
        return np.nan

    if pair[target].std() < 1e-12:
        return np.nan

    try:

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )

            result = grangercausalitytests(
                pair,
                maxlag=lag,
                verbose=False
            )

        # SSR F-test p-value.
        p_value = result[
            lag
        ][0]["ssr_ftest"][1]

        return p_value

    except Exception:

        return np.nan


# ============================================================
# RUN ANALYSIS
# ============================================================

results = []

faults = ground_truth[
    "fault"
].unique()

print("\nFaults:")
for fault in faults:
    print(f"  {fault}")


for _, fault_row in ground_truth.iterrows():

    run_id = int(
        fault_row["run_id"]
    )

    fault_name = fault_row[
        "fault"
    ]

    start = float(
        fault_row["start_time"]
    )

    end = float(
        fault_row["end_time"]
    )

    data = get_fault_series(
        run_id,
        start,
        end
    )

    if len(data) < MIN_SAMPLES:
        continue

    print(
        f"\nRun {run_id} | "
        f"{fault_name} | "
        f"{start:.1f}-{end:.1f} ms | "
        f"{len(data)} samples"
    )

    # --------------------------------------------------------
    # Every ordered pair of components.
    # --------------------------------------------------------

    for source in QUEUE_COLUMNS:

        for target in QUEUE_COLUMNS:

            if source == target:
                continue

            for lag in TEST_LAGS:

                p_value = run_granger_test(
                    source,
                    target,
                    data,
                    lag
                )

                results.append({
                    "run_id": run_id,
                    "fault": fault_name,
                    "fault_start": start,
                    "fault_end": end,
                    "source": source,
                    "target": target,
                    "lag_ms": lag,
                    "p_value": p_value,
                    "significant": (
                        p_value < 0.05
                        if not np.isnan(p_value)
                        else False
                    )
                })


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRANGER CAUSALITY SUMMARY")
print("=" * 70)

print(
    f"Tests performed: {len(results_df)}"
)

valid = results_df[
    results_df["p_value"].notna()
]

print(
    f"Valid tests: {len(valid)}"
)

print(
    f"Significant tests (p < 0.05): "
    f"{valid['significant'].sum()}"
)


# ------------------------------------------------------------
# Most frequently significant relationships
# ------------------------------------------------------------

if len(valid) > 0:

    summary = (
        valid
        .groupby(
            ["fault", "source", "target"]
        )
        .agg(
            tests=("p_value", "count"),
            significant=(
                "significant",
                "sum"
            ),
            min_p_value=(
                "p_value",
                "min"
            )
        )
        .reset_index()
    )

    summary[
        "significant_fraction"
    ] = (
        summary["significant"]
        /
        summary["tests"]
    )

    summary = summary.sort_values(
        [
            "fault",
            "significant_fraction",
            "min_p_value"
        ],
        ascending=[
            True,
            False,
            True
        ]
    )

    print("\nTop relationships by fault:")

    for fault in faults:

        fault_summary = summary[
            summary["fault"] == fault
        ]

        print(
            f"\n--- {fault} ---"
        )

        print(
            fault_summary.head(10)[
                [
                    "source",
                    "target",
                    "tests",
                    "significant",
                    "significant_fraction",
                    "min_p_value"
                ]
            ].to_string(
                index=False
            )
        )


print("\nSaved:")
print(OUTPUT_PATH)