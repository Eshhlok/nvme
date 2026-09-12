import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_PATH = "output/granger_windows.csv"
OUTPUT_PATH = "output/granger_fault_windows.png"

QUEUE_COLUMNS = [
    "host_cpu_q",
    "host_pcie_q",
    "host_nic_q",
    "network_q",
    "target_nic_q",
    "target_pcie_q",
    "ssd_q",
]

FAULTS = [
    "cpu_contention",
    "pcie_contention",
    "network_congestion",
    "noisy_neighbour",
    "ssd_saturation",
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print(f"Rows loaded: {len(df)}")


# ============================================================
# SELECT ONE REPRESENTATIVE RUN PER FAULT
# ============================================================

selected = []

for fault in FAULTS:

    fault_df = df[
        df["fault"] == fault
    ]

    # Use the first available run for each fault.
    run_id = fault_df["run_id"].iloc[0]

    segment = fault_df[
        fault_df["run_id"] == run_id
    ].copy()

    selected.append(segment)

    print(
        f"{fault}: Run {run_id}, "
        f"{len(segment)} samples"
    )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    len(FAULTS),
    1,
    figsize=(14, 18),
    sharex=True
)


for ax, fault, segment in zip(
    axes,
    FAULTS,
    selected
):

    for queue in QUEUE_COLUMNS:

        ax.plot(
            segment["relative_time"],
            segment[queue],
            label=queue,
            linewidth=1
        )

    # Fault starts at relative time = 0.
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.5
    )

    # Fault ends approximately 600 ms after onset.
    ax.axvline(
        600,
        linestyle="--",
        linewidth=1.5
    )

    ax.set_title(
        fault.replace("_", " ").title()
    )

    ax.set_ylabel(
        "Queue depth"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="upper right",
        fontsize=7,
        ncol=4
    )


axes[-1].set_xlabel(
    "Time relative to fault start (ms)"
)

fig.suptitle(
    "Queue Dynamics Around Fault Intervals",
    fontsize=16,
    fontweight="bold"
)

fig.text(
    0.5,
    0.01,
    "Dashed lines indicate fault onset (0 ms) and nominal fault end (~600 ms)",
    ha="center",
    fontsize=10
)

plt.tight_layout(
    rect=[0, 0.025, 1, 0.97]
)


# ============================================================
# SAVE
# ============================================================

Path(
    OUTPUT_PATH
).parent.mkdir(
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nSaved: {OUTPUT_PATH}"
)