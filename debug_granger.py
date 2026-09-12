import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests


INPUT_PATH = "output/granger_windows.csv"

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


df = pd.read_csv(INPUT_PATH)

# Take the first fault interval
group = df.groupby(
    ["run_id", "fault", "fault_start", "fault_end"]
)

(name, segment) = next(iter(group))

run_id, fault, fault_start, fault_end = name

segment = segment.sort_values("relative_time")

print("=" * 70)
print("GRANGER DEBUG")
print("=" * 70)
print(f"Run   : {run_id}")
print(f"Fault : {fault}")
print(f"Rows  : {len(segment)}")
print()


for source in QUEUE_COLUMNS:

    for target in QUEUE_COLUMNS:

        if source == target:
            continue

        source_diff = (
            segment[source]
            .astype(float)
            .diff()
            .dropna()
        )

        target_diff = (
            segment[target]
            .astype(float)
            .diff()
            .dropna()
        )

        data = pd.DataFrame({
            "target": target_diff,
            "source": source_diff
        }).dropna()

        if data["source"].std() < 1e-12:
            continue

        if data["target"].std() < 1e-12:
            continue

        print(
            f"\n{source} -> {target}"
        )

        print(
            f"source std = {data['source'].std():.6f}, "
            f"target std = {data['target'].std():.6f}"
        )

        for lag in LAGS:

            try:

                result = grangercausalitytests(
                    data[["target", "source"]],
                    maxlag=lag,
                    verbose=False
                )

                p = result[lag][0]["ssr_ftest"][1]

                print(
                    f"  lag={lag:2d} ms -> "
                    f"SUCCESS, p={p:.6g}"
                )

            except Exception as e:

                print(
                    f"  lag={lag:2d} ms -> "
                    f"ERROR: {type(e).__name__}: {e}"
                )