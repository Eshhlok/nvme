import pandas as pd
from pathlib import Path


INPUT_PATH = "output/granger_results_fdr.csv"
OUTPUT_PATH = "output/granger_relationship_summary.csv"

FAULTS = [
    "cpu_contention",
    "pcie_contention",
    "network_congestion",
    "noisy_neighbour",
    "ssd_saturation",
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=== GRANGER RELATIONSHIP SUMMARY ===")
print(f"Rows loaded: {len(df)}")


# ============================================================
# ONLY FDR-SIGNIFICANT RESULTS
# ============================================================

sig = df[
    df["significant_fdr"]
].copy()

print(
    f"FDR-significant tests: {len(sig)}"
)


# ============================================================
# AGGREGATE BY FAULT + SOURCE + TARGET + LAG
# ============================================================
#
# Each fault occurs exactly 30 times.
#
# Therefore:
#
# significant_count = number of runs in which
#                     this relationship was significant
#
# ============================================================

summary = (
    sig
    .groupby(
        [
            "fault",
            "source",
            "target",
            "lag_ms"
        ]
    )
    .agg(
        significant_runs=(
            "run_id",
            "nunique"
        )
    )
    .reset_index()
)

summary["total_runs"] = 30

summary["significant_rate"] = (
    summary["significant_runs"] /
    summary["total_runs"]
)


# ============================================================
# SORT
# ============================================================

summary = summary.sort_values(
    [
        "fault",
        "significant_runs",
        "lag_ms"
    ],
    ascending=[
        True,
        False,
        True
    ]
)


# ============================================================
# SAVE
# ============================================================

Path(
    OUTPUT_PATH
).parent.mkdir(
    exist_ok=True
)

summary.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# PRINT TOP RELATIONSHIPS PER FAULT
# ============================================================

for fault in FAULTS:

    print("\n" + "=" * 70)
    print(fault.upper())
    print("=" * 70)

    fault_summary = summary[
        summary["fault"] == fault
    ]

    if len(fault_summary) == 0:

        print("No FDR-significant relationships.")

        continue

    print(
        fault_summary
        .head(15)
        .to_string(index=False)
    )


# ============================================================
# BEST RELATIONSHIP PER SOURCE → TARGET
# ============================================================
#
# A relationship can be significant at multiple lags.
# This gives us the strongest observed lag for each
# source → target pair.
# ============================================================

best = (
    summary
    .sort_values(
        [
            "fault",
            "source",
            "target",
            "significant_runs",
            "lag_ms"
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True
        ]
    )
    .drop_duplicates(
        subset=[
            "fault",
            "source",
            "target"
        ]
    )
)


# ============================================================
# MOST CONSISTENT RELATIONSHIPS OVERALL
# ============================================================

print("\n" + "=" * 70)
print("MOST CONSISTENT RELATIONSHIPS")
print("=" * 70)

print(
    best
    .sort_values(
        "significant_runs",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


print(
    f"\nSaved: {OUTPUT_PATH}"
)