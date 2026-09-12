import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    "output/"
    "final_xgboost_fault_localization_model.joblib"
)

INPUT_PATH = (
    "output/"
    "all_fault_queues.csv"
)

OUTPUT_PATH = (
    "output/"
    "end_to_end_localization_results.csv"
)


WINDOW_SIZE = 100


# ============================================================
# LOAD MODEL
# ============================================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_cols = bundle["features"]
class_names = bundle["class_names"]


# ============================================================
# QUEUE COLUMNS
# ============================================================

QUEUE_COLUMNS = [
    "host_cpu_q",
    "host_pcie_q",
    "host_nic_q",
    "network_q",
    "target_nic_q",
    "target_pcie_q",
    "ssd_q",
]


# ============================================================
# FAULT → PHYSICAL LOCATION
# ============================================================

FAULT_LOCATION = {

    "baseline":
        "No fault detected",

    "cpu_contention":
        "Host CPU",

    "pcie_contention":
        "Host PCIe",

    "network_congestion":
        "Network",

    "noisy_neighbour":
        "Target NIC / Target PCIe",

    "ssd_saturation":
        "SSD",
}


# ============================================================
# TEMPORAL FEATURE
# ============================================================

def max_consecutive_nonzero(series):

    values = (
        series
        .fillna(0)
        .values
    )

    maximum = 0
    current = 0

    for value in values:

        if value != 0:
            current += 1
            maximum = max(
                maximum,
                current
            )

        else:
            current = 0

    return maximum


# ============================================================
# EXTRACT 9 FEATURES FOR ONE QUEUE
# ============================================================

def extract_queue_features(series):

    series = (
        series
        .fillna(0)
        .astype(float)
    )

    mean_value = series.mean()

    return {

        "mean":
            mean_value,

        "max":
            series.max(),

        "std":
            series.std(),

        "nonzero":
            (series != 0).mean(),

        "p95":
            series.quantile(0.95),

        "p99":
            series.quantile(0.99),

        "max_nonzero_run":
            max_consecutive_nonzero(series),

        "transitions":
            (series.diff().fillna(0) != 0).sum(),

        "above_mean":
            (series > mean_value).mean(),
    }


# ============================================================
# BUILD 63-FEATURE WINDOWS
# ============================================================

def build_features(queues):

    queues = queues.copy()

    queues["window_start"] = (
        np.floor(
            queues["time"] /
            WINDOW_SIZE
        ) *
        WINDOW_SIZE
    )

    queues["window_start"] = (
        queues["window_start"]
        .astype(float)
    )

    feature_rows = []

    grouped = queues.groupby(
        ["run_id", "window_start"],
        sort=True
    )

    for (
        (run_id, window_start),
        window
    ) in grouped:

        row = {

            "run_id":
                run_id,

            "window_start":
                window_start,

            "window_end":
                window_start +
                WINDOW_SIZE,
        }

        for queue in QUEUE_COLUMNS:

            features = extract_queue_features(
                window[queue]
            )

            for suffix, value in features.items():

                row[
                    f"{queue}_{suffix}"
                ] = value

        feature_rows.append(row)

    return pd.DataFrame(
        feature_rows
    )


# ============================================================
# LOCALIZE
# ============================================================

def localize(features):

    X = features[
        feature_cols
    ]

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    results = features[
        [
            "run_id",
            "window_start",
            "window_end"
        ]
    ].copy()

    results["predicted_fault"] = [
        class_names[int(prediction)]
        for prediction in predictions
    ]

    results["latency_source"] = [
        FAULT_LOCATION[
            class_names[int(prediction)]
        ]
        for prediction in predictions
    ]

    results["confidence"] = [
        probabilities[i][int(predictions[i])]
        for i in range(len(predictions))
    ]

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("END-TO-END NVMe LATENCY SOURCE LOCALIZATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Load raw queue telemetry
    # --------------------------------------------------------

    queues = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"\nRaw telemetry rows: "
        f"{len(queues)}"
    )

    print(
        f"Runs: "
        f"{queues['run_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    features = build_features(
        queues
    )

    print(
        f"Feature windows: "
        f"{len(features)}"
    )

    print(
        f"Features generated: "
        f"{len(feature_cols)}"
    )

    # --------------------------------------------------------
    # Verify feature compatibility
    # --------------------------------------------------------

    missing = [
        col
        for col in feature_cols
        if col not in features.columns
    ]

    if missing:

        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Localization
    # --------------------------------------------------------

    results = localize(
        features
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    Path(
        OUTPUT_PATH
    ).parent.mkdir(
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("LOCALIZATION SUMMARY")
    print("=" * 65)

    print(
        results[
            "predicted_fault"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Show representative examples
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("REPRESENTATIVE LOCALIZATION RESULTS")
    print("=" * 65)

    examples = []

    for fault in class_names:

        matching = results[
            results["predicted_fault"] == fault
        ]

        if len(matching) > 0:

            examples.append(
                matching.iloc[0]
            )

    for row in examples:

        print("\n" + "-" * 55)

        print(
            f"Run             : "
            f"{int(row['run_id'])}"
        )

        print(
            f"Window          : "
            f"{row['window_start']:.0f} - "
            f"{row['window_end']:.0f} ms"
        )

        print(
            f"Predicted fault : "
            f"{row['predicted_fault']}"
        )

        print(
            f"Latency source  : "
            f"{row['latency_source']}"
        )

        print(
            f"Confidence      : "
            f"{row['confidence']:.2%}"
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )