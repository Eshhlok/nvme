import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "output/gnn_dataset.pt"
RESULTS_PATH = "output/gnn_v2_cv_results.csv"
OOF_PATH = "output/gnn_v2_oof_predictions.csv"

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
# MODEL
# ============================================================

class GCNV2(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Keep both average behaviour and strongest local signal.
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)

        graph_embedding = torch.cat(
            [mean_pool, max_pool],
            dim=1
        )

        graph_embedding = self.dropout(graph_embedding)

        return self.classifier(graph_embedding)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

data_bundle = torch.load(
    DATASET_PATH,
    weights_only=False
)

dataset = data_bundle["graphs"]
class_names = data_bundle["class_names"]

print(f"Graphs: {len(dataset)}")

run_ids = np.array([
    int(data.run_id)
    for data in dataset
])

labels = np.array([
    int(data.y.item())
    for data in dataset
])

num_classes = len(np.unique(labels))
input_dim = dataset[0].x.shape[1]

print(f"Nodes per graph: {dataset[0].x.shape[0]}")
print(f"Features per node: {input_dim}")
print(f"Classes: {num_classes}")


# ============================================================
# CLASS NAMES
# ============================================================


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_datasets(train_dataset, val_dataset):

    # Stack every node from every training graph.
    train_nodes = torch.cat(
        [data.x for data in train_dataset],
        dim=0
    )

    mean = train_nodes.mean(dim=0)
    std = train_nodes.std(dim=0)

    # Prevent division by zero.
    std[std < 1e-8] = 1.0

    train_normalized = []

    for data in train_dataset:

        new_data = copy.deepcopy(data)

        new_data.x = (new_data.x - mean) / std

        train_normalized.append(new_data)

    val_normalized = []

    for data in val_dataset:

        new_data = copy.deepcopy(data)

        new_data.x = (new_data.x - mean) / std

        val_normalized.append(new_data)

    return train_normalized, val_normalized


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(train_dataset, val_dataset):

    model = GCNV2(
        input_dim=input_dim,
        hidden_dim=HIDDEN,
        num_classes=num_classes
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

    best_model = None
    best_val_loss = float("inf")
    best_epoch = 0

    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:

            optimizer.zero_grad()

            out = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            loss = criterion(out, batch.y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

            predictions = out.argmax(dim=1)

            train_correct += (
                predictions == batch.y
            ).sum().item()

            train_total += batch.num_graphs

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

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

                val_loss += (
                    loss.item() * batch.num_graphs
                )

                predictions = out.argmax(dim=1)

                val_correct += (
                    predictions == batch.y
                ).sum().item()

                val_total += batch.num_graphs

        val_loss /= val_total
        val_acc = val_correct / val_total

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            best_model = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"Val Acc {val_acc:.4f}"
            )

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch}"
            )

            break

    model.load_state_dict(best_model)

    print(
        f"Best epoch: {best_epoch} | "
        f"Best validation loss: {best_val_loss:.4f}"
    )

    return model


# ============================================================
# 5-FOLD GROUP CROSS VALIDATION
# ============================================================

print("\nStarting 5-fold GroupKFold...\n")

gkf = GroupKFold(
    n_splits=N_SPLITS
)

results = []
oof_predictions = []

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(
        dataset,
        labels,
        groups=run_ids
    ),
    start=1
):

    print("=" * 70)
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

    train_runs = sorted(
        set(run_ids[train_idx])
    )

    val_runs = sorted(
        set(run_ids[val_idx])
    )

    print(f"Train runs: {train_runs}")
    print(f"Validation runs: {val_runs}")

    # --------------------------------------------------------
    # Normalize using TRAINING graphs only.
    # --------------------------------------------------------

    train_dataset, val_dataset = normalize_datasets(
        train_dataset,
        val_dataset
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_fold(
        train_dataset,
        val_dataset
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model.eval()

    fold_true = []
    fold_pred = []

    with torch.no_grad():

        for batch in loader:

            out = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            predictions = out.argmax(
                dim=1
            )

            fold_true.extend(
                batch.y.cpu().numpy()
            )

            fold_pred.extend(
                predictions.cpu().numpy()
            )

    fold_acc = accuracy_score(
        fold_true,
        fold_pred
    )

    fold_f1 = f1_score(
        fold_true,
        fold_pred,
        average="macro"
    )

    print(
        f"\nFold {fold} Accuracy: "
        f"{fold_acc:.4f}"
    )

    print(
        f"Fold {fold} Macro F1: "
        f"{fold_f1:.4f}"
    )

    results.append({
        "fold": fold,
        "accuracy": fold_acc,
        "macro_f1": fold_f1
    })

    # --------------------------------------------------------
    # Store OOF predictions
    # --------------------------------------------------------

    for idx, true_label, pred_label in zip(
        val_idx,
        fold_true,
        fold_pred
    ):

        oof_predictions.append({
            "run_id": int(run_ids[idx]),
            "window_start": float(
                dataset[idx].window_start
            ),
            "window_end": float(
                dataset[idx].window_end
            ),
            "true_label": class_names[true_label],
            "predicted_label": class_names[pred_label]
        })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

mean_accuracy = results_df["accuracy"].mean()
std_accuracy = results_df["accuracy"].std()

mean_f1 = results_df["macro_f1"].mean()
std_f1 = results_df["macro_f1"].std()

print("\n" + "=" * 70)
print("FINAL GNN V2 RESULTS")
print("=" * 70)

print(
    f"Mean Accuracy : "
    f"{mean_accuracy:.4f} "
    f"(std {std_accuracy:.4f})"
)

print(
    f"Mean Macro F1 : "
    f"{mean_f1:.4f} "
    f"(std {std_f1:.4f})"
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

oof_df = pd.DataFrame(
    oof_predictions
)

oof_df.to_csv(
    OOF_PATH,
    index=False
)

print("\nSaved:")
print(RESULTS_PATH)
print(OOF_PATH)