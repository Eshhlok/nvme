import pandas as pd
import torch

from torch_geometric.data import Data


INPUT_FILE = "output/overlap_fault_features.csv"
OUTPUT_FILE = "output/gnn_dataset.pt"


# --------------------------------------------------
# Load feature dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Dataset:")
print(f"  Rows: {len(df)}")
print(f"  Runs: {df['run_id'].nunique()}")


# --------------------------------------------------
# Node definitions
# --------------------------------------------------

nodes = [
    "host_cpu",
    "host_pcie",
    "host_nic",
    "network",
    "target_nic",
    "target_pcie",
    "ssd"
]


# --------------------------------------------------
# Features for every node
# --------------------------------------------------

feature_suffixes = [
    "mean",
    "max",
    "std",
    "nonzero",
    "p95",
    "p99",
    "max_nonzero_run",
    "transitions",
    "above_mean"
]


# --------------------------------------------------
# Graph topology
#
# CPU ↔ PCIe ↔ Host NIC ↔ Network
#      ↔ Target NIC ↔ Target PCIe ↔ SSD
#
# Actual chain:
#
# CPU ↔ Host PCIe ↔ Host NIC ↔ Network
#      ↔ Target NIC ↔ Target PCIe ↔ SSD
# --------------------------------------------------

edges = [
    (0, 1),
    (1, 0),

    (1, 2),
    (2, 1),

    (2, 3),
    (3, 2),

    (3, 4),
    (4, 3),

    (4, 5),
    (5, 4),

    (5, 6),
    (6, 5)
]

edge_index = torch.tensor(
    edges,
    dtype=torch.long
).t().contiguous()


# --------------------------------------------------
# Label encoding
# --------------------------------------------------

class_names = sorted(df["label"].unique())

label_to_id = {
    label: i
    for i, label in enumerate(class_names)
}

print("\nClasses:")
for label, idx in label_to_id.items():
    print(f"  {idx}: {label}")


# --------------------------------------------------
# Build graphs
# --------------------------------------------------

graphs = []

for _, row in df.iterrows():

    node_features = []

    for node in nodes:

        features = []

        for suffix in feature_suffixes:

            column = f"{node}_q_{suffix}"

            features.append(
                float(row[column])
            )

        node_features.append(features)

    x = torch.tensor(
        node_features,
        dtype=torch.float32
    )

    y = torch.tensor(
        label_to_id[row["label"]],
        dtype=torch.long
    )

    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    # Keep metadata for later analysis
    graph.run_id = int(row["run_id"])
    graph.window_start = float(row["window_start"])
    graph.window_end = float(row["window_end"])

    graphs.append(graph)


# --------------------------------------------------
# Save dataset
# --------------------------------------------------

torch.save(
    {
        "graphs": graphs,
        "class_names": class_names,
        "nodes": nodes,
        "feature_suffixes": feature_suffixes
    },
    OUTPUT_FILE
)


# --------------------------------------------------
# Verification
# --------------------------------------------------

print("\n=== GNN DATASET ===")

print(f"Graphs       : {len(graphs)}")
print(f"Nodes/graph  : {graphs[0].x.shape[0]}")
print(f"Features/node: {graphs[0].x.shape[1]}")
print(f"Edges/graph  : {graphs[0].edge_index.shape[1]}")

print("\nFirst graph:")
print(graphs[0])

print("\nNode feature matrix shape:")
print(graphs[0].x.shape)

print("\nEdge index:")
print(graphs[0].edge_index)

print("\nFirst graph label:")
print(
    graphs[0].y.item(),
    "->",
    class_names[graphs[0].y.item()]
)

print("\nSaved:")
print(OUTPUT_FILE)