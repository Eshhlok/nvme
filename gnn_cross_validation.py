import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score


DATASET_FILE = "output/gnn_dataset.pt"

N_SPLITS = 5
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_DIM = 32

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------
# Load graph dataset
# --------------------------------------------------

dataset_package = torch.load(
    DATASET_FILE,
    weights_only=False
)

graphs = dataset_package["graphs"]
class_names = dataset_package["class_names"]

print("=== GNN DATASET ===")
print(f"Graphs       : {len(graphs)}")
print(f"Nodes/graph  : {graphs[0].x.shape[0]}")
print(f"Features/node: {graphs[0].x.shape[1]}")
print(f"Classes      : {class_names}")


# --------------------------------------------------
# Metadata for GroupKFold
# --------------------------------------------------

run_ids = np.array([
    graph.run_id
    for graph in graphs
])

labels = np.array([
    graph.y.item()
    for graph in graphs
])


# --------------------------------------------------
# GCN model
# --------------------------------------------------

class FaultGCN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):

        x = self.conv1(
            x,
            edge_index
        )

        x = self.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = self.relu(x)

        # Convert node representations
        # into one representation per graph
        x = global_mean_pool(
            x,
            batch
        )

        return self.classifier(x)


# --------------------------------------------------
# Normalize node features
# --------------------------------------------------

def normalize_graphs(
    train_graphs,
    test_graphs
):

    # Stack all training node features
    train_features = torch.cat(
        [
            graph.x
            for graph in train_graphs
        ],
        dim=0
    )

    mean = train_features.mean(
        dim=0,
        keepdim=True
    )

    std = train_features.std(
        dim=0,
        keepdim=True
    )

    # Prevent division by zero
    std[std < 1e-8] = 1.0

    normalized_train = []

    for graph in train_graphs:

        graph_copy = graph.clone()

        graph_copy.x = (
            graph_copy.x - mean
        ) / std

        normalized_train.append(
            graph_copy
        )

    normalized_test = []

    for graph in test_graphs:

        graph_copy = graph.clone()

        graph_copy.x = (
            graph_copy.x - mean
        ) / std

        normalized_test.append(
            graph_copy
        )

    return normalized_train, normalized_test


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_model(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    for batch in loader:

        optimizer.zero_grad()

        output = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = criterion(
            output,
            batch.y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * batch.num_graphs
        )

    return total_loss / len(loader.dataset)


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate(
    model,
    loader
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

            output = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            predictions = output.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch.y.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return accuracy, macro_f1, all_predictions


# --------------------------------------------------
# 5-fold GroupKFold
# --------------------------------------------------

cv = GroupKFold(
    n_splits=N_SPLITS
)

fold_accuracies = []
fold_f1_scores = []

oof_predictions = np.full(
    len(graphs),
    -1,
    dtype=int
)


for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        graphs,
        labels,
        groups=run_ids
    ),
    start=1
):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    train_graphs = [
        graphs[i]
        for i in train_idx
    ]

    test_graphs = [
        graphs[i]
        for i in test_idx
    ]

    # ----------------------------------------------
    # Verify run-level separation
    # ----------------------------------------------

    train_runs = sorted(
        set(
            run_ids[train_idx]
        )
    )

    test_runs = sorted(
        set(
            run_ids[test_idx]
        )
    )

    assert set(train_runs).isdisjoint(
        set(test_runs)
    )

    print(
        f"Train runs : {train_runs}"
    )

    print(
        f"Test runs  : {test_runs}"
    )

    # ----------------------------------------------
    # Normalize using training data ONLY
    # ----------------------------------------------

    train_graphs, test_graphs = normalize_graphs(
        train_graphs,
        test_graphs
    )

    # ----------------------------------------------
    # Data loaders
    # ----------------------------------------------

    train_loader = DataLoader(
        train_graphs,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ----------------------------------------------
    # Model
    # ----------------------------------------------

    model = FaultGCN(
        input_dim=9,
        hidden_dim=HIDDEN_DIM,
        num_classes=len(class_names)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------
    # Train
    # ----------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        loss = train_model(
            model,
            train_loader,
            optimizer,
            criterion
        )

        if (
            epoch == 1
            or epoch % 20 == 0
            or epoch == EPOCHS
        ):

            accuracy, macro_f1, _ = evaluate(
                model,
                test_loader
            )

            print(
                f"Epoch {epoch:3d} | "
                f"Loss {loss:.4f} | "
                f"Val Acc {accuracy:.4f} | "
                f"Val F1 {macro_f1:.4f}"
            )

    # ----------------------------------------------
    # Final fold evaluation
    # ----------------------------------------------

    accuracy, macro_f1, predictions = evaluate(
        model,
        test_loader
    )

    fold_accuracies.append(
        accuracy
    )

    fold_f1_scores.append(
        macro_f1
    )

    oof_predictions[test_idx] = predictions

    print(
        f"\nFold {fold} Final:"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro F1 : {macro_f1:.4f}"
    )


# --------------------------------------------------
# Overall results
# --------------------------------------------------

mean_accuracy = np.mean(
    fold_accuracies
)

std_accuracy = np.std(
    fold_accuracies
)

mean_f1 = np.mean(
    fold_f1_scores
)

std_f1 = np.std(
    fold_f1_scores
)


print("\n")
print("=" * 60)
print("GNN CROSS-VALIDATION RESULTS")
print("=" * 60)

for i in range(N_SPLITS):

    print(
        f"Fold {i + 1}: "
        f"Accuracy={fold_accuracies[i]:.4f}, "
        f"Macro F1={fold_f1_scores[i]:.4f}"
    )

print(
    f"\nMean Accuracy : {mean_accuracy:.4f}"
)

print(
    f"Std Accuracy  : {std_accuracy:.4f}"
)

print(
    f"Mean Macro F1 : {mean_f1:.4f}"
)

print(
    f"Std Macro F1  : {std_f1:.4f}"
)


# --------------------------------------------------
# Save CV results
# --------------------------------------------------

results = pd.DataFrame({
    "fold": range(1, N_SPLITS + 1),
    "accuracy": fold_accuracies,
    "macro_f1": fold_f1_scores
})

results.to_csv(
    "output/gnn_cv_results.csv",
    index=False
)


# --------------------------------------------------
# Save OOF predictions
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "run_id": run_ids,
    "window_start": [
        graph.window_start
        for graph in graphs
    ],
    "window_end": [
        graph.window_end
        for graph in graphs
    ],
    "label": [
        class_names[label]
        for label in labels
    ],
    "prediction": [
        class_names[prediction]
        for prediction in oof_predictions
    ]
})

prediction_df.to_csv(
    "output/gnn_oof_predictions.csv",
    index=False
)


print("\nSaved:")
print("output/gnn_cv_results.csv")
print("output/gnn_oof_predictions.csv")