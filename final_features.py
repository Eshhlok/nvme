import pandas as pd
from sklearn.ensemble import RandomForestClassifier

INPUT_FILE = "output/overlap_fault_features.csv"

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# Select queue features
# --------------------------------------------------

feature_cols = [
    col for col in df.columns
    if "_q_" in col
]

X = df[feature_cols]
y = df["label"]

print("Queue features:", len(feature_cols))
print("Samples:", len(X))

# --------------------------------------------------
# Train final Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X, y)

# --------------------------------------------------
# Feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

print("\n=== TOP 25 FEATURES ===")

print(
    importance.head(25).to_string(index=False)
)

# --------------------------------------------------
# Group importance by queue
# --------------------------------------------------

queue_importance = {}

for feature, value in zip(
    importance["feature"],
    importance["importance"]
):
    queue = feature.split("_q_")[0]

    queue_importance[queue] = (
        queue_importance.get(queue, 0) + value
    )

queue_importance_df = (
    pd.DataFrame(
        list(queue_importance.items()),
        columns=["queue", "importance"]
    )
    .sort_values("importance", ascending=False)
)

print("\n=== IMPORTANCE BY QUEUE ===")

print(
    queue_importance_df.to_string(index=False)
)

# --------------------------------------------------
# Save results
# --------------------------------------------------

importance.to_csv(
    "output/final_feature_importance.csv",
    index=False
)

queue_importance_df.to_csv(
    "output/final_queue_importance.csv",
    index=False
)

print("\nSaved:")
print("output/final_feature_importance.csv")
print("output/final_queue_importance.csv")