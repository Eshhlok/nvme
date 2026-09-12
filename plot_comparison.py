import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv("output/baseline_queues.csv")
fault = pd.read_csv("output/pcie_contention_queues.csv")

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# PCIe
axes[0].plot(
    base["time"],
    base["host_pcie_q"],
    label="baseline",
    linewidth=2,
    zorder=2
)

axes[0].plot(
    fault["time"],
    fault["host_pcie_q"],
    label="pcie_contention",
    linewidth=1,
    alpha=0.8,
    zorder=3
)

axes[0].set_title("PCIe queue depth")
axes[0].legend()

# Network
axes[1].plot(
    base["time"],
    base["network_q"],
    label="baseline",
    linewidth=2,
    zorder=2
)

axes[1].plot(
    fault["time"],
    fault["network_q"],
    label="pcie_contention",
    linewidth=1,
    alpha=0.8,
    zorder=3
)

axes[1].set_title("Network queue depth")
axes[1].legend()

plt.tight_layout()
plt.savefig("output/pcie_fingerprint.png")
print("Saved output/pcie_fingerprint.png")