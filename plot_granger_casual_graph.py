import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


INPUT_PATH = "output/granger_relationship_summary.csv"
OUTPUT_PATH = "output/granger_causal_graphs.png"

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


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=== GRANGER CAUSAL GRAPH ===")
print(f"Rows loaded: {len(df)}")


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 11)
)

axes = axes.flatten()


for ax, fault in zip(axes, FAULTS):

    fault_df = df[
        (df["fault"] == fault) &
        (df["significant_runs"] >= MIN_RUNS)
    ].copy()

    # --------------------------------------------------------
    # Keep strongest lag for each source -> target pair
    # --------------------------------------------------------

    fault_df = (
        fault_df
        .sort_values(
            ["source", "target", "significant_runs", "lag_ms"],
            ascending=[True, True, False, True]
        )
        .drop_duplicates(
            subset=["source", "target"]
        )
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    G = nx.DiGraph()

    G.add_nodes_from(NODE_ORDER)

    for _, row in fault_df.iterrows():

        G.add_edge(
            row["source"],
            row["target"],
            runs=row["significant_runs"],
            rate=row["significant_rate"],
            lag=row["lag_ms"]
        )

    # --------------------------------------------------------
    # Position nodes according to pipeline
    # --------------------------------------------------------

    pos = {
        "host_cpu_q": (0, 0),
        "host_pcie_q": (1, 0),
        "host_nic_q": (2, 0),
        "network_q": (3, 0),
        "target_nic_q": (4, 0),
        "target_pcie_q": (5, 0),
        "ssd_q": (6, 0),
    }

    # --------------------------------------------------------
    # Draw nodes
    # --------------------------------------------------------

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=1800
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=NODE_LABELS,
        ax=ax,
        font_size=8
    )

    # --------------------------------------------------------
    # Draw edges
    # --------------------------------------------------------

    if len(G.edges) > 0:

        widths = []

        for _, _, data in G.edges(data=True):

            widths.append(
                1.5 + 4 * data["rate"]
            )

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=widths,
            arrows=True,
            arrowsize=18,
            connectionstyle="arc3,rad=0.12"
        )

        # ----------------------------------------------------
        # Edge labels
        # ----------------------------------------------------

        edge_labels = {}

        for source, target, data in G.edges(data=True):

            edge_labels[
                (source, target)
            ] = (
                f"{data['runs']}/30\n"
                f"{data['lag']} ms"
            )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=7,
            label_pos=0.5
        )

    ax.set_title(
        fault.replace("_", " ").title(),
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlim(-0.6, 6.6)
    ax.set_ylim(-1.2, 1.2)

    ax.axis("off")


# Hide unused sixth panel
axes[-1].axis("off")


fig.suptitle(
    "FDR-Significant Granger Causal Relationships",
    fontsize=17,
    fontweight="bold"
)

fig.text(
    0.5,
    0.02,
    "Only relationships significant in at least "
    f"{MIN_RUNS}/30 runs are shown. "
    "Edge label = significant runs / representative lag.",
    ha="center",
    fontsize=10
)

plt.tight_layout(
    rect=[0, 0.05, 1, 0.95]
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