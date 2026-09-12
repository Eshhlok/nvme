import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv("output/baseline_queues.csv")
fault = pd.read_csv("output/ssd_saturation_queues.csv")

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# SSD
axes[0].plot(
    base["time"],
    base["ssd_q"],
    label="baseline",
    linewidth=2,
    zorder=2
)

axes[0].plot(
    fault["time"],
    fault["ssd_q"],
    label="ssd_saturation",
    linewidth=1,
    alpha=0.8,
    zorder=3
)

axes[0].set_title("SSD queue depth")
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
    label="ssd_saturation",
    linewidth=1,
    alpha=0.8,
    zorder=3
)

axes[1].set_title("Network queue depth")
axes[1].legend()

plt.tight_layout()
plt.savefig("output/ssd_fingerprint.png")

print("Saved output/ssd_fingerprint.png")