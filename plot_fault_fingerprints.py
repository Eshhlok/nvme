import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "output/overlap_fault_features.csv"
OUTPUT_FILE = "output/fault_queue_fingerprints.png"

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# Queue mean features
queue_features = [
    "host_cpu_q_mean",
    "host_pcie_q_mean",
    "host_nic_q_mean",
    "network_q_mean",
    "target_nic_q_mean",
    "target_pcie_q_mean",
    "ssd_q_mean"
]

fault_order = [
    "baseline",
    "cpu_contention",
    "pcie_contention",
    "network_congestion",
    "noisy_neighbour",
    "ssd_saturation"
]

# --------------------------------------------------
# Calculate mean queue depth per fault
# --------------------------------------------------

fingerprints = (
    df.groupby("label")[queue_features]
    .mean()
    .reindex(fault_order)
)

# Shorter names for plot
fingerprints.columns = [
    "Host CPU",
    "Host PCIe",
    "Host NIC",
    "Network",
    "Target NIC",
    "Target PCIe",
    "SSD"
]

# --------------------------------------------------
# Plot
# --------------------------------------------------

ax = fingerprints.plot(
    kind="bar",
    figsize=(13, 7)
)

ax.set_title(
    "NVMe Queue Telemetry Fingerprints by Fault Type"
)

ax.set_xlabel("Fault Type")
ax.set_ylabel("Mean Queue Depth")

plt.xticks(rotation=25, ha="right")
plt.legend(
    title="Pipeline Queue",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {OUTPUT_FILE}")