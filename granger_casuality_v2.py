import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests


INPUT_PATH = "output/granger_windows.csv"
OUTPUT_PATH = "output/granger_results_v2.csv"

QUEUE_COLUMNS = [
    "host_cpu_q",
    "host_pcie_q",
    "host_nic_q",
    "network_q",
    "target_nic_q",
    "target_pcie_q",
    "ssd_q",
]

LAGS = [1, 5, 10, 20]
ALPHA = 0.05


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=== GRANGER CAUSALITY V2 ===")
print(f"Rows loaded: {len(df)}")
print(f"Fault intervals: {df['run_id'].nunique() * 5}")


# ============================================================
# RESULTS
# ============================================================

results = []

fault_groups = df.groupby(
    ["run_id", "fault", "fault_start", "fault_end"],
    sort=True
)

total_intervals = len(fault_groups)

print(f"Intervals to analyze: {total_intervals}")


# ============================================================
# RUN GRANGER TESTS
# ============================================================

for interval_idx, (
    (run_id, fault, fault_start, fault_end),
    segment
) in enumerate(fault_groups, start=1):

    segment = segment.sort_values("relative_time")

    print(
        f"\n[{interval_idx}/{total_intervals}] "
        f"Run {run_id} | {fault}"
    )

    for source in QUEUE_COLUMNS:

        for target in QUEUE_COLUMNS:

            if source == target:
                continue

            # ------------------------------------------------
            # Extract the two time series
            # ------------------------------------------------

            source_series = segment[source].astype(float)
            target_series = segment[target].astype(float)

            # ------------------------------------------------
            # First difference
            # ------------------------------------------------
            #
            # This focuses on queue changes rather than
            # absolute queue levels.
            #
            # Example:
            # 0, 0, 0, 2, 4, 6, 6, 6
            #
            # becomes approximately:
            # 0, 0, 2, 2, 2, 0, 0
            #
            # which is much more useful for temporal analysis.
            # ------------------------------------------------

            source_diff = source_series.diff().dropna()
            target_diff = target_series.diff().dropna()

            data = pd.DataFrame({
                "target": target_diff,
                "source": source_diff
            }).dropna()

            # ------------------------------------------------
            # Basic validity checks
            # ------------------------------------------------

            if len(data) < max(LAGS) + 20:
                continue

            if data["source"].std() < 1e-12:
                continue

            if data["target"].std() < 1e-12:
                continue

            # ------------------------------------------------
            # Test each lag
            # ------------------------------------------------

            for lag in LAGS:

                try:

                    test_result = grangercausalitytests(
                        data[["target", "source"]],
                        maxlag=[lag]
                        
                    )

                    # statsmodels returns results for every
                    # lag from 1 through maxlag.
                    #
                    # We only want the requested lag itself.

                    result = test_result[lag]

                    # SSR F-test
                    p_value = result[0]["ssr_ftest"][1]

                    results.append({
                        "run_id": run_id,
                        "fault": fault,
                        "fault_start": fault_start,
                        "fault_end": fault_end,
                        "source": source,
                        "target": target,
                        "lag_ms": lag,
                        "p_value": p_value,
                        "significant": p_value < ALPHA,
                    })

                except Exception as e:

                    results.append({
                        "run_id": run_id,
                        "fault": fault,
                        "fault_start": fault_start,
                        "fault_end": fault_end,
                        "source": source,
                        "target": target,
                        "lag_ms": lag,
                        "p_value": np.nan,
                        "significant": False,
                    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

Path(
    OUTPUT_PATH
).parent.mkdir(
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRANGER SUMMARY")
print("=" * 70)

print(
    f"Tests performed : {len(results_df)}"
)

print(
    f"Valid tests     : "
    f"{results_df['p_value'].notna().sum()}"
)

print(
    f"Significant     : "
    f"{results_df['significant'].sum()}"
)


# ============================================================
# SIGNIFICANT RELATIONSHIPS
# ============================================================

sig = results_df[
    results_df["significant"]
].copy()

if len(sig) == 0:

    print("\nNo significant relationships found.")

else:

    print(
        "\n=== MOST FREQUENT SIGNIFICANT RELATIONSHIPS ==="
    )

    summary = (
        sig
        .groupby(
            ["fault", "source", "target", "lag_ms"]
        )
        .size()
        .reset_index(name="significant_count")
        .sort_values(
            "significant_count",
            ascending=False
        )
    )

    print(
        summary.head(30).to_string(index=False)
    )


# ============================================================
# SIGNIFICANCE RATE
# ============================================================

print(
    "\n=== SIGNIFICANCE RATE BY FAULT ==="
)

fault_summary = (
    results_df
    .groupby("fault")
    .agg(
        tests=("p_value", "count"),
        valid=("p_value", lambda x: x.notna().sum()),
        significant=("significant", "sum")
    )
)

fault_summary["significance_rate"] = (
    fault_summary["significant"] /
    fault_summary["valid"]
)

print(
    fault_summary.to_string()
)


print(
    f"\nSaved: {OUTPUT_PATH}"
)