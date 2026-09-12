import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv("output/baseline_queues.csv")
fault = pd.read_csv("output/noisy_neighbour_queues.csv")

fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

# Target NIC
axes[0].plot(
    base["time"],
    base["target_nic_q"],
    label="baseline",
    linewidth=2
)

axes[0].plot(
    fault["time"],
    fault["target_nic_q"],
    label="noisy_neighbour",
    linewidth=1,
    alpha=0.8
)

axes[0].set_title("Target NIC queue depth")
axes[0].legend()

# Target PCIe
axes[1].plot(
    base["time"],
    base["target_pcie_q"],
    label="baseline",
    linewidth=2
)

axes[1].plot(
    fault["time"],
    fault["target_pcie_q"],
    label="noisy_neighbour",
    linewidth=1,
    alpha=0.8
)

axes[1].set_title("Target PCIe queue depth")
axes[1].legend()

# SSD
axes[2].plot(
    base["time"],
    base["ssd_q"],
    label="baseline",
    linewidth=2
)

axes[2].plot(
    fault["time"],
    fault["ssd_q"],
    label="noisy_neighbour",
    linewidth=1,
    alpha=0.8
)

axes[2].set_title("SSD queue depth")
axes[2].legend()

plt.tight_layout()
plt.savefig("output/noisy_neighbour_fingerprint.png")

print("Saved output/noisy_neighbour_fingerprint.png")