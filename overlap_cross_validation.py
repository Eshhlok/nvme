import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score


FEATURE_FILE = "output/overlap_fault_features.csv"
OUTPUT_FILE = "output/overlap_cross_validation_results.csv"

df = pd.read_csv(FEATURE_FILE)

queue_features = [
    col for col in df.columns
    if "_q_" in col
]

X = df[queue_features]
y = df["label"]
groups = df["run_id"]

print("========================================")
print("50% OVERLAP LABELING - QUEUE FEATURE CV")
print("========================================")
print("Rows:", len(df))
print("Queue features:", len(queue_features))
print("Runs:", df["run_id"].nunique())

print("\nClass distribution:")
print(y.value_counts())


group_kfold = GroupKFold(n_splits=5)

fold_results = []

for fold, (train_idx, test_idx) in enumerate(
    group_kfold.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    fold_results.append({
        "fold": fold,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    print(f"\nFold {fold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


results_df = pd.DataFrame(fold_results)

mean_accuracy = results_df["accuracy"].mean()
std_accuracy = results_df["accuracy"].std()

mean_f1 = results_df["macro_f1"].mean()
std_f1 = results_df["macro_f1"].std()


print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print(f"Mean Accuracy : {mean_accuracy:.4f}")
print(f"Std Accuracy  : {std_accuracy:.4f}")

print(f"Mean Macro F1 : {mean_f1:.4f}")
print(f"Std Macro F1  : {std_f1:.4f}")

print("\nPrevious enhanced result:")
print("Mean Accuracy : 0.9087")
print("Mean Macro F1 : 0.9173")

print("\nOriginal baseline:")
print("Mean Accuracy : 0.9080")
print("Mean Macro F1 : 0.9165")

print("\nChange vs enhanced:")
print(
    f"Accuracy: {(mean_accuracy - 0.9087) * 100:+.2f} pp"
)

print(
    f"Macro F1: {(mean_f1 - 0.9173) * 100:+.2f} pp"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_FILE)