import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold

INPUT_FILE = "output/overlap_fault_features.csv"
OUTPUT_FILE = "output/final_confusion_matrix.png"

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

feature_cols = [
    col for col in df.columns
    if "_q_" in col
]

X = df[feature_cols]
y = df["label"]
groups = df["run_id"]

# --------------------------------------------------
# Generate out-of-fold predictions
# --------------------------------------------------

cv = GroupKFold(n_splits=5)

predictions = pd.Series(
    index=df.index,
    dtype=object
)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    predictions.iloc[test_idx] = model.predict(
        X.iloc[test_idx]
    )

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

labels = [
    "baseline",
    "cpu_contention",
    "network_congestion",
    "noisy_neighbour",
    "pcie_contention",
    "ssd_saturation"
]

cm = confusion_matrix(
    y,
    predictions,
    labels=labels
)

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Baseline",
        "CPU",
        "Network",
        "Noisy\nNeighbour",
        "PCIe",
        "SSD"
    ]
)

display.plot(
    ax=ax,
    values_format="d",
    cmap="Blues",
    colorbar=False
)

ax.set_title(
    "Fault Localization — 5-Fold Run-Level Cross-Validation"
)

ax.set_xlabel("Predicted Fault")
ax.set_ylabel("Actual Fault")

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {OUTPUT_FILE}")