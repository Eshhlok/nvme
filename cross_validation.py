import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("output/all_fault_features.csv")

print("Dataset loaded")
print("Rows:", len(df))
print("Runs:", df["run_id"].nunique())


# --------------------------------------------------
# 2. Select queue features only
# --------------------------------------------------

queue_features = [
    col for col in df.columns
    if "_q_" in col
]

X = df[queue_features]
y = df["label"]
groups = df["run_id"]


print("Queue features:", len(queue_features))


# --------------------------------------------------
# 3. Group-based cross validation
# --------------------------------------------------
# Each entire simulation run stays in either
# training OR testing.
#
# This prevents windows from the same run
# appearing in both sets.

gkf = GroupKFold(n_splits=5)


results = []


# --------------------------------------------------
# 4. Train and evaluate each fold
# --------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_runs = sorted(groups.iloc[train_idx].unique())
    test_runs = sorted(groups.iloc[test_idx].unique())

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    print("Training runs:", train_runs)
    print("Testing runs :", test_runs)

    print("Training rows:", len(train_idx))
    print("Testing rows :", len(test_idx))


    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")


    results.append({
        "fold": fold,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })


# --------------------------------------------------
# 5. Results summary
# --------------------------------------------------

results_df = pd.DataFrame(results)

mean_accuracy = results_df["accuracy"].mean()
std_accuracy = results_df["accuracy"].std()

mean_f1 = results_df["macro_f1"].mean()
std_f1 = results_df["macro_f1"].std()


print("\n")
print("=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

print("\nFold results:")
print(
    results_df.to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "macro_f1": "{:.4f}".format
        }
    )
)

print("\nMean Accuracy:")
print(f"{mean_accuracy:.4f} ({mean_accuracy * 100:.2f}%)")

print("Accuracy Std Dev:")
print(f"{std_accuracy:.4f}")

print("\nMean Macro F1:")
print(f"{mean_f1:.4f}")

print("Macro F1 Std Dev:")
print(f"{std_f1:.4f}")


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

results_df.to_csv(
    "output/cross_validation_results.csv",
    index=False
)

print("\nResults saved to:")
print("output/cross_validation_results.csv")