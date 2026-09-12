import pandas as pd
from statsmodels.stats.multitest import multipletests


INPUT_PATH = "output/granger_results_v2.csv"
OUTPUT_PATH = "output/granger_results_fdr.csv"

ALPHA = 0.05


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=== FDR-CORRECTED GRANGER ANALYSIS ===")
print(f"Rows loaded: {len(df)}")


# ============================================================
# REMOVE INVALID TESTS
# ============================================================

valid = df["p_value"].notna()

results = df[valid].copy()

print(f"Valid tests: {len(results)}")


# ============================================================
# BENJAMINI-HOCHBERG FDR CORRECTION
# ============================================================
#
# We correct all valid Granger tests together.
#
# This controls the expected false-discovery rate rather
# than treating every raw p < 0.05 as independently reliable.
# ============================================================

reject, p_adjusted, _, _ = multipletests(
    results["p_value"].values,
    alpha=ALPHA,
    method="fdr_bh"
)

results["p_value_fdr"] = p_adjusted
results["significant_fdr"] = reject


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

raw_significant = (
    results["p_value"] < ALPHA
).sum()

fdr_significant = (
    results["significant_fdr"]
).sum()

print(
    f"Raw significant (p < 0.05) : "
    f"{raw_significant}"
)

print(
    f"FDR significant            : "
    f"{fdr_significant}"
)

print(
    f"Raw significance rate      : "
    f"{raw_significant / len(results):.2%}"
)

print(
    f"FDR significance rate      : "
    f"{fdr_significant / len(results):.2%}"
)


# ============================================================
# FAULT-WISE SUMMARY
# ============================================================

print("\n=== SIGNIFICANCE BY FAULT ===")

fault_summary = (
    results
    .groupby("fault")
    .agg(
        tests=("p_value", "count"),
        raw_significant=(
            "p_value",
            lambda x: (x < ALPHA).sum()
        ),
        fdr_significant=(
            "significant_fdr",
            "sum"
        )
    )
)

fault_summary["raw_rate"] = (
    fault_summary["raw_significant"] /
    fault_summary["tests"]
)

fault_summary["fdr_rate"] = (
    fault_summary["fdr_significant"] /
    fault_summary["tests"]
)

print(
    fault_summary.to_string()
)


# ============================================================
# MOST CONSISTENT RELATIONSHIPS
# ============================================================

print("\n=== MOST CONSISTENT FDR-SIGNIFICANT RELATIONSHIPS ===")

sig = results[
    results["significant_fdr"]
].copy()

if len(sig) == 0:

    print("No relationships survived FDR correction.")

else:

    relationship_summary = (
        sig
        .groupby(
            ["fault", "source", "target", "lag_ms"]
        )
        .size()
        .reset_index(
            name="significant_count"
        )
        .sort_values(
            [
                "fault",
                "significant_count"
            ],
            ascending=[True, False]
        )
    )

    print(
        relationship_summary
        .head(50)
        .to_string(index=False)
    )


print(
    f"\nSaved: {OUTPUT_PATH}"
)