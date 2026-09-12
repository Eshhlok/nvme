import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool
)

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "output/gnn_dataset.pt"
OUTPUT_PATH = "output/gnn_node_attribution.csv"

N_SPLITS = 5
HIDDEN = 64
LR = 0.001
EPOCHS = 150
BATCH_SIZE = 32
DROPOUT = 0.30
PATIENCE = 15
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# NODE NAMES
# ============================================================

NODE_NAMES = [
    "host_cpu",
    "host_pcie",
    "host_nic",
    "network",
    "target_nic",
    "target_pcie",
    "ssd",
]


# ============================================================
# MODEL
# ============================================================

class GCNV2(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_classes
        )

    def forward(self, x, edge_index, batch):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        mean_pool = global_mean_pool(
            x,
            batch
        )

        max_pool = global_max_pool(
            x,
            batch
        )

        embedding = torch.cat(
            [mean_pool, max_pool],
            dim=1
        )

        embedding = self.dropout(
            embedding
        )

        return self.classifier(
            embedding
        )


# ============================================================
# LOAD DATA
# ============================================================

bundle = torch.load(
    DATASET_PATH,
    weights_only=False
)

dataset = bundle["graphs"]
class_names = bundle["class_names"]

run_ids = np.array([
    int(data.run_id)
    for data in dataset
])

labels = np.array([
    int(data.y.item())
    for data in dataset
])

input_dim = dataset[0].x.shape[1]
num_classes = len(class_names)

print(f"Graphs: {len(dataset)}")
print(f"Nodes: {dataset[0].x.shape[0]}")
print(f"Features per node: {input_dim}")
print(f"Classes: {num_classes}")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_datasets(train_dataset, val_dataset):

    train_nodes = torch.cat(
        [data.x for data in train_dataset],
        dim=0
    )

    mean = train_nodes.mean(dim=0)
    std = train_nodes.std(dim=0)

    std[std < 1e-8] = 1.0

    train_norm = []

    for data in train_dataset:

        new_data = copy.deepcopy(data)

        new_data.x = (
            new_data.x - mean
        ) / std

        train_norm.append(new_data)

    val_norm = []

    for data in val_dataset:

        new_data = copy.deepcopy(data)

        new_data.x = (
            new_data.x - mean
        ) / std

        val_norm.append(new_data)

    return train_norm, val_norm, mean, std


# ============================================================
# TRAIN GNN
# ============================================================

def train_model(train_dataset, val_dataset):

    model = GCNV2(
        input_dim,
        HIDDEN,
        num_classes
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    best_state = None
    best_loss = float("inf")
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        for batch in train_loader:

            optimizer.zero_grad()

            out = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            loss = criterion(
                out,
                batch.y
            )

            loss.backward()
            optimizer.step()

        # Validation loss
        model.eval()

        total_loss = 0.0
        total_graphs = 0

        with torch.no_grad():

            for batch in val_loader:

                out = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch
                )

                loss = criterion(
                    out,
                    batch.y
                )

                total_loss += (
                    loss.item() *
                    batch.num_graphs
                )

                total_graphs += batch.num_graphs

        val_loss = total_loss / total_graphs

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_count = 0

        else:

            patience_count += 1

        if patience_count >= PATIENCE:
            break

    model.load_state_dict(best_state)

    return model


# ============================================================
# SINGLE GRAPH PREDICTION
# ============================================================

def predict_graph(model, graph):

    model.eval()

    loader = DataLoader(
        [graph],
        batch_size=1,
        shuffle=False
    )

    with torch.no_grad():

        batch = next(iter(loader))

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    return probabilities.cpu()


# ============================================================
# NODE OCCLUSION
# ============================================================

def calculate_node_attribution(
    model,
    graph,
    baseline_node_features
):

    original_prob = predict_graph(
        model,
        graph
    )

    predicted_class = int(
        original_prob.argmax()
    )

    original_confidence = float(
        original_prob[predicted_class]
    )

    node_importance = {}

    for node_idx, node_name in enumerate(
        NODE_NAMES
    ):

        modified_graph = copy.deepcopy(
            graph
        )

        # Replace this node's complete
        # 9-dimensional feature vector.
        modified_graph.x[node_idx] = (
            baseline_node_features
        )

        modified_prob = predict_graph(
            model,
            modified_graph
        )

        # Drop in confidence for the
        # original prediction.
        importance = (
            original_confidence -
            float(
                modified_prob[predicted_class]
            )
        )

        node_importance[node_name] = importance

    most_influential = max(
        node_importance,
        key=node_importance.get
    )

    return (
        predicted_class,
        original_confidence,
        most_influential,
        node_importance
    )


# ============================================================
# CROSS-VALIDATED ATTRIBUTION
# ============================================================

gkf = GroupKFold(
    n_splits=N_SPLITS
)

records = []

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(
        dataset,
        labels,
        groups=run_ids
    ),
    start=1
):

    print("\n" + "=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    train_dataset = [
        dataset[i]
        for i in train_idx
    ]

    val_dataset = [
        dataset[i]
        for i in val_idx
    ]

    (
        train_dataset,
        val_dataset,
        train_mean,
        train_std
    ) = normalize_datasets(
        train_dataset,
        val_dataset
    )

    # Mean feature vector used for node occlusion.
    # Since data are normalized, zero corresponds
    # to the training mean.
    baseline_node_features = torch.zeros(
        input_dim
    )

    model = train_model(
        train_dataset,
        val_dataset
    )

    correct = 0

    for local_idx, graph in enumerate(
        val_dataset
    ):

        (
            predicted_class,
            confidence,
            influential_node,
            importance
        ) = calculate_node_attribution(
            model,
            graph,
            baseline_node_features
        )

        true_class = int(
            graph.y.item()
        )

        if predicted_class == true_class:
            correct += 1

        record = {
            "fold": fold,
            "run_id": int(graph.run_id),
            "window_start": float(
                graph.window_start
            ),
            "window_end": float(
                graph.window_end
            ),
            "true_label": class_names[
                true_class
            ],
            "predicted_label": class_names[
                predicted_class
            ],
            "prediction_confidence": confidence,
            "most_influential_node": influential_node
        }

        for node_name in NODE_NAMES:

            record[
                f"{node_name}_importance"
            ] = importance[node_name]

        records.append(record)

    fold_accuracy = (
        correct /
        len(val_dataset)
    )

    print(
        f"Fold {fold} accuracy: "
        f"{fold_accuracy:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("NODE ATTRIBUTION COMPLETE")
print("=" * 70)

print(
    f"Graphs analyzed: {len(df)}"
)

print(
    f"Output: {OUTPUT_PATH}"
)

print("\nMost influential node counts:")

print(
    df[
        "most_influential_node"
    ].value_counts()
)

print("\nCorrect predictions only:")

correct_df = df[
    df["true_label"] ==
    df["predicted_label"]
]

print(
    correct_df[
        "most_influential_node"
    ].value_counts()
)