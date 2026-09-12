import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


INPUT_PATH = "output/granger_relationship_summary.csv"
OUTPUT_PATH = "output/final_granger_causal_graphs.png"

MIN_RUNS = 15

FAULTS = [
    "cpu_contention",
    "pcie_contention",
    "network_congestion",
    "noisy_neighbour",
    "ssd_saturation",
]

NODE_ORDER = [
    "host_cpu_q",
    "host_pcie_q",
    "host_nic_q",
    "network_q",
    "target_nic_q",
    "target_pcie_q",
    "ssd_q",
]

NODE_LABELS = {
    "host_cpu_q": "Host CPU",
    "host_pcie_q": "Host PCIe",
    "host_nic_q": "Host NIC",
    "network_q": "Network",
    "target_nic_q": "Target NIC",
    "target_pcie_q": "Target PCIe",
    "ssd_q": "SSD",
}

NODE_POS = {
    "host_cpu_q": (0, 0),
    "host_pcie_q": (1, 0),
    "host_nic_q": (2, 0),
    "network_q": (3, 0),
    "target_nic_q": (4, 0),
    "target_pcie_q": (5, 0),
    "ssd_q": (6, 0),
}


df = pd.read_csv(INPUT_PATH)

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10)
)

axes = axes.flatten()


for ax, fault in zip(axes, FAULTS):

    fault_df = df[
        (df["fault"] == fault) &
        (df["significant_runs"] >= MIN_RUNS)
    ].copy()

    # Keep strongest lag for each direction.
    fault_df = (
        fault_df
        .sort_values(
            ["source", "target", "significant_runs", "lag_ms"],
            ascending=[True, True, False, True]
        )
        .drop_duplicates(
            ["source", "target"]
        )
    )

    G = nx.DiGraph()

    G.add_nodes_from(NODE_ORDER)

    for _, row in fault_df.iterrows():

        G.add_edge(
            row["source"],
            row["target"],
            runs=int(row["significant_runs"]),
            rate=row["significant_rate"],
            lag=int(row["lag_ms"])
        )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    nx.draw_networkx_nodes(
        G,
        NODE_POS,
        ax=ax,
        node_size=1500
    )

    nx.draw_networkx_labels(
        G,
        NODE_POS,
        labels=NODE_LABELS,
        ax=ax,
        font_size=8
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    if len(G.edges) > 0:

        edge_widths = [
            1.0 + 4.0 * data["rate"]
            for _, _, data in G.edges(data=True)
        ]

        nx.draw_networkx_edges(
            G,
            NODE_POS,
            ax=ax,
            width=edge_widths,
            arrows=True,
            arrowsize=16,
            node_size=1500,
            connectionstyle="arc3,rad=0.18"
        )

        edge_labels = {}

        for source, target, data in G.edges(data=True):

            edge_labels[
                (source, target)
            ] = (
                f"{data['rate']:.0%}\n"
                f"{data['lag']} ms"
            )

        nx.draw_networkx_edge_labels(
            G,
            NODE_POS,
            edge_labels=edge_labels,
            ax=ax,
            font_size=7,
            label_pos=0.5,
            rotate=False
        )

    # --------------------------------------------------------
    # Titles
    # --------------------------------------------------------

    ax.set_title(
        fault.replace("_", " ").title(),
        fontsize=12,
        fontweight="bold"
    )

    if len(G.edges) == 0:

        ax.text(
            3,
            0,
            "No relationship ≥ 50% of runs",
            ha="center",
            va="center",
            fontsize=10
        )

    ax.set_xlim(-0.7, 6.7)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")


# Hide unused sixth panel.
axes[-1].axis("off")


fig.suptitle(
    "FDR-Corrected Granger Temporal Relationships",
    fontsize=17,
    fontweight="bold"
)

fig.text(
    0.5,
    0.035,
    "Edges appear only when significant in ≥15/30 runs. "
    "Label = repeatability and representative lag. "
    "Thicker edges indicate higher repeatability.",
    ha="center",
    fontsize=10
)

plt.tight_layout(
    rect=[0, 0.07, 1, 0.94]
)

Path(
    OUTPUT_PATH
).parent.mkdir(
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=220,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {OUTPUT_PATH}")