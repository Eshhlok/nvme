import pandas as pd
import numpy as np

FILE = "output/all_fault_features.csv"

df = pd.read_csv(FILE)

print("=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

# --------------------------------------------------
# 1. Basic information
# --------------------------------------------------

print("\n1. BASIC INFORMATION")
print("-" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Runs:", df["run_id"].nunique())

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. Missing values
# --------------------------------------------------

print("\n2. MISSING VALUES")
print("-" * 60)

missing = df.isnull().sum()

print(
    missing[missing > 0]
    if missing.sum() > 0
    else "No missing values"
)


# --------------------------------------------------
# 3. Infinite values
# --------------------------------------------------

print("\n3. INFINITE VALUES")
print("-" * 60)

numeric_df = df.select_dtypes(include=np.number)

infinite_count = np.isinf(numeric_df).sum().sum()

print("Infinite values:", infinite_count)


# --------------------------------------------------
# 4. Class distribution
# --------------------------------------------------

print("\n4. CLASS DISTRIBUTION")
print("-" * 60)

print(df["label"].value_counts())

print("\nClass percentages:")
print(
    (df["label"].value_counts(normalize=True) * 100)
    .round(2)
)


# --------------------------------------------------
# 5. Distribution across runs
# --------------------------------------------------

print("\n5. CLASS DISTRIBUTION BY RUN")
print("-" * 60)

run_class = pd.crosstab(
    df["run_id"],
    df["label"]
)

print(run_class)


# --------------------------------------------------
# 6. Feature summary
# --------------------------------------------------

print("\n6. FEATURE SUMMARY")
print("-" * 60)

feature_columns = [
    column
    for column in df.columns
    if column not in [
        "run_id",
        "window_start",
        "window_end",
        "label"
    ]
]

print(
    df[feature_columns].describe().T
)


# --------------------------------------------------
# 7. Mean feature values by fault
# --------------------------------------------------

print("\n7. MEAN FEATURES BY CLASS")
print("-" * 60)

class_means = df.groupby("label")[feature_columns].mean()

print(
    class_means.round(3).T
)


# --------------------------------------------------
# 8. Feature variance
# --------------------------------------------------

print("\n8. ZERO-VARIANCE FEATURES")
print("-" * 60)

zero_variance = []

for column in feature_columns:

    if df[column].nunique() <= 1:
        zero_variance.append(column)

if zero_variance:
    print(zero_variance)
else:
    print("No zero-variance features")


# --------------------------------------------------
# 9. Highly correlated features
# --------------------------------------------------

print("\n9. HIGHLY CORRELATED FEATURES")
print("-" * 60)

corr = df[feature_columns].corr()

high_corr_pairs = []

for i in range(len(corr.columns)):

    for j in range(i + 1, len(corr.columns)):

        value = corr.iloc[i, j]

        if abs(value) >= 0.95:

            high_corr_pairs.append(
                (
                    corr.columns[i],
                    corr.columns[j],
                    value
                )
            )

if high_corr_pairs:

    for feature1, feature2, value in high_corr_pairs:
        print(
            f"{feature1} <-> {feature2}: "
            f"{value:.3f}"
        )

else:
    print("No feature pairs with correlation >= 0.95")


# --------------------------------------------------
# 10. Train/test split by RUN
# --------------------------------------------------

print("\n10. RUN-LEVEL SPLIT")
print("-" * 60)

train_runs = list(range(1, 25))
test_runs = list(range(25, 31))

train_df = df[
    df["run_id"].isin(train_runs)
]

test_df = df[
    df["run_id"].isin(test_runs)
]

print("Training runs:", train_runs)
print("Testing runs:", test_runs)

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nTraining class distribution:")
print(train_df["label"].value_counts())

print("\nTesting class distribution:")
print(test_df["label"].value_counts())


# --------------------------------------------------
# 11. Final check
# --------------------------------------------------

print("\n11. FINAL CHECK")
print("-" * 60)

print(
    "Expected training rows: 1200"
)

print(
    "Expected testing rows: 300"
)

print(
    "Actual training rows:",
    len(train_df)
)

print(
    "Actual testing rows:",
    len(test_df)
)

print("\nInspection complete.")