import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("output/all_fault_features.csv")

print("Dataset loaded")
print("Total rows:", len(df))


# --------------------------------------------------
# 2. Train / test split by RUN
# --------------------------------------------------
# Runs 1-24 -> training
# Runs 25-30 -> testing
#
# This is important because we want to test on
# completely unseen simulation runs.

train_df = df[df["run_id"] <= 24].copy()
test_df = df[df["run_id"] >= 25].copy()

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nTraining runs:", sorted(train_df["run_id"].unique()))
print("Testing runs:", sorted(test_df["run_id"].unique()))


# --------------------------------------------------
# 3. Separate features and labels
# --------------------------------------------------

drop_columns = [
    "run_id",
    "window_start",
    "window_end",
    "label"
]

X_train = train_df.drop(columns=drop_columns)
y_train = train_df["label"]

X_test = test_df.drop(columns=drop_columns)
y_test = test_df["label"]


print("\nNumber of features:", X_train.shape[1])


# --------------------------------------------------
# 4. Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete.")


# --------------------------------------------------
# 5. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 6. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 7. Classification report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 8. Confusion matrix
# --------------------------------------------------

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nConfusion Matrix:")
print(cm_df)


# --------------------------------------------------
# 9. Feature importance
# --------------------------------------------------

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n==============================")
print("TOP 15 FEATURE IMPORTANCES")
print("==============================")

print(importance_df.head(15).to_string(index=False))


# --------------------------------------------------
# 10. Save trained model
# --------------------------------------------------

model_path = "output/random_forest_model.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")


# --------------------------------------------------
# 11. Save predictions
# --------------------------------------------------

predictions = test_df[
    ["run_id", "window_start", "window_end", "label"]
].copy()

predictions["predicted_label"] = y_pred

predictions.to_csv(
    "output/test_predictions.csv",
    index=False
)

print("Predictions saved to: output/test_predictions.csv")