import pandas as pd
import matplotlib.pyplot as plt

q = pd.read_csv("output/random_faults_queues.csv")
lat = pd.read_csv("output/random_faults_latency.csv")
gt = pd.read_csv("output/random_faults_ground_truth.csv")


fig, axes = plt.subplots(
    7, 1,
    figsize=(12, 14),
    sharex=True
)


# -------------------------
# Fault periods
# -------------------------

faults = [
    (row["start_time"], row["end_time"], row["fault"])
    for _, row in gt.iterrows()
]


def mark_faults(ax):
    for start, end, fault in faults:
        ax.axvspan(
            start,
            end,
            alpha=0.15
        )

        ax.text(
            (start + end) / 2,
            0.95,
            fault,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8
        )


# -------------------------
# Queue plots
# -------------------------

queues = [
    ("host_cpu_q", "Host CPU queue"),
    ("host_pcie_q", "Host PCIe queue"),
    ("network_q", "Network queue"),
    ("target_nic_q", "Target NIC queue"),
    ("target_pcie_q", "Target PCIe queue"),
    ("ssd_q", "SSD queue"),
]


for ax, (column, title) in zip(axes[:6], queues):

    ax.plot(
        q["time"],
        q[column],
        linewidth=1
    )

    ax.set_ylabel("Queue")
    ax.set_title(title)

    mark_faults(ax)


# -------------------------
# Latency
# -------------------------

axes[6].plot(
    lat["finish_time"],
    lat["latency_ms"],
    marker=".",
    linestyle="None",
    markersize=3
)

axes[6].set_title("Request latency")
axes[6].set_xlabel("Simulation time (ms)")
axes[6].set_ylabel("Latency (ms)")

mark_faults(axes[6])


plt.tight_layout()

plt.savefig(
    "output/random_faults_fingerprint.png",
    dpi=150
)

print("Saved output/random_faults_fingerprint.png")