import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score


INPUT_FILE = "output/overlap_fault_features.csv"

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
# Encode labels
# --------------------------------------------------

class_names = sorted(y.unique())

label_to_id = {
    label: i
    for i, label in enumerate(class_names)
}

y_encoded = y.map(label_to_id)


# --------------------------------------------------
# Small controlled parameter set
# --------------------------------------------------

configs = [
    {
        "name": "baseline",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1
    },
    {
        "name": "shallower",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1
    },
    {
        "name": "deeper",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "min_child_weight": 1
    },
    {
        "name": "slower_learning",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "min_child_weight": 1
    },
    {
        "name": "stronger_leaf",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 3
    },
    {
        "name": "stronger_leaf_deeper",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "min_child_weight": 3
    }
]


# --------------------------------------------------
# Same GroupKFold for every configuration
# --------------------------------------------------

cv = GroupKFold(n_splits=5)

results = []


# --------------------------------------------------
# Evaluate each configuration
# --------------------------------------------------

for config in configs:

    print("\n" + "=" * 60)
    print(f"CONFIGURATION: {config['name']}")
    print("=" * 60)

    accuracies = []
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y_encoded, groups),
        start=1
    ):

        model = XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            min_child_weight=config["min_child_weight"],
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=len(class_names),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X.iloc[train_idx],
            y_encoded.iloc[train_idx]
        )

        predictions = model.predict(
            X.iloc[test_idx]
        )

        accuracy = accuracy_score(
            y_encoded.iloc[test_idx],
            predictions
        )

        macro_f1 = f1_score(
            y_encoded.iloc[test_idx],
            predictions,
            average="macro"
        )

        accuracies.append(accuracy)
        f1_scores.append(macro_f1)

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro F1={macro_f1:.4f}"
        )

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

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

    results.append({
        "configuration": config["name"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "min_child_weight": config["min_child_weight"],
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1
    })


# --------------------------------------------------
# Compare configurations
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mean_macro_f1",
    ascending=False
)

print("\n")
print("=" * 75)
print("XGBOOST TUNING RESULTS")
print("=" * 75)

print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

results_df.to_csv(
    "output/xgboost_tuning_results.csv",
    index=False
)

print(
    "\nSaved: output/xgboost_tuning_results.csv"
)