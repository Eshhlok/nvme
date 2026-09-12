import pandas as pd
import matplotlib.pyplot as plt

QUEUE_FILE = "output/all_fault_queues.csv"
GROUND_TRUTH_FILE = "output/all_fault_ground_truth.csv"

RUN_ID = 1
OUTPUT_FILE = "output/fault_timeseries_run1.png"

# --------------------------------------------------
# Load data
# --------------------------------------------------

queues = pd.read_csv(QUEUE_FILE)
faults = pd.read_csv(GROUND_TRUTH_FILE)

run_queues = queues[
    queues["run_id"] == RUN_ID
].copy()

run_faults = faults[
    faults["run_id"] == RUN_ID
].copy()

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

queue_names = {
    "host_cpu_q": "Host CPU",
    "host_pcie_q": "Host PCIe",
    "host_nic_q": "Host NIC",
    "network_q": "Network",
    "target_nic_q": "Target NIC",
    "target_pcie_q": "Target PCIe",
    "ssd_q": "SSD"
}

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(15, 7))

for column in queue_columns:

    ax.plot(
        run_queues["time"],
        run_queues[column],
        label=queue_names[column],
        linewidth=1.2
    )

# --------------------------------------------------
# Mark fault intervals
# --------------------------------------------------

for _, fault in run_faults.iterrows():

    start = fault["start_time"]
    end = fault["end_time"]

    ax.axvspan(
        start,
        end,
        alpha=0.12
    )

    ax.text(
        (start + end) / 2,
        ax.get_ylim()[1] * 0.92,
        fault["fault"],
        ha="center",
        va="top",
        rotation=90,
        fontsize=8
    )

# --------------------------------------------------
# Labels
# --------------------------------------------------

ax.set_title(
    f"NVMe Queue Behavior Across Faults — Run {RUN_ID}"
)

ax.set_xlabel("Simulation Time (ms)")
ax.set_ylabel("Queue Depth")

ax.legend(
    title="Pipeline Queue",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {OUTPUT_FILE}")