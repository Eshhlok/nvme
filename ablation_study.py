import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("output/all_fault_features.csv")

train_df = df[df["run_id"] <= 24].copy()
test_df = df[df["run_id"] >= 25].copy()

y_train = train_df["label"]
y_test = test_df["label"]


# --------------------------------------------------
# 2. Define feature groups
# --------------------------------------------------

queue_features = [
    col for col in df.columns
    if "_q_" in col
]

latency_features = [
    "latency_mean",
    "latency_max",
    "latency_p95",
    "latency_std"
]


# --------------------------------------------------
# 3. Function to train and evaluate
# --------------------------------------------------

def evaluate_model(feature_columns, name):

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print(f"\n{name}")
    print("-" * 40)
    print("Number of features:", len(feature_columns))
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")

    return accuracy, macro_f1


# --------------------------------------------------
# 4. Run experiments
# --------------------------------------------------

results = []


# All features
all_features = queue_features + latency_features

accuracy, f1 = evaluate_model(
    all_features,
    "ALL FEATURES"
)

results.append([
    "All features",
    len(all_features),
    accuracy,
    f1
])


# Queue features only
accuracy, f1 = evaluate_model(
    queue_features,
    "QUEUE FEATURES ONLY"
)

results.append([
    "Queue features only",
    len(queue_features),
    accuracy,
    f1
])


# Latency features only
accuracy, f1 = evaluate_model(
    latency_features,
    "LATENCY FEATURES ONLY"
)

results.append([
    "Latency features only",
    len(latency_features),
    accuracy,
    f1
])


# --------------------------------------------------
# 5. Summary
# --------------------------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Feature Set",
        "Number of Features",
        "Accuracy",
        "Macro F1"
    ]
)

print("\n")
print("=" * 60)
print("ABLATION STUDY SUMMARY")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Macro F1": "{:.4f}".format
        }
    )
)


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

results_df.to_csv(
    "output/ablation_results.csv",
    index=False
)

print("\nResults saved to:")
print("output/ablation_results.csv")