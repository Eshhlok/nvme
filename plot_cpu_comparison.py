import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv("output/baseline_queues.csv")
fault = pd.read_csv("output/cpu_contention_queues.csv")

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(base["time"], base["host_cpu_q"], label="baseline")
axes[0].plot(fault["time"], fault["host_cpu_q"], label="cpu_contention")
axes[0].set_title("Host CPU queue depth (should be elevated)")
axes[0].legend()

axes[1].plot(base["time"], base["host_pcie_q"], label="baseline")
axes[1].plot(fault["time"], fault["host_pcie_q"], label="cpu_contention")
axes[1].set_title("PCIe queue depth (should stay roughly flat)")
axes[1].legend()

plt.tight_layout()
plt.savefig("output/cpu_fingerprint.png")
print("Saved output/cpu_fingerprint.png")
