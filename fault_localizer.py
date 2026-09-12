import pandas as pd
import joblib


MODEL_PATH = (
    "output/"
    "final_xgboost_fault_localization_model.joblib"
)


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
# LOAD FINAL MODEL
# ============================================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_cols = bundle["features"]
class_names = bundle["class_names"]


# ============================================================
# FAULT LOCALIZATION
# ============================================================

def localize_fault(feature_row):

    X = pd.DataFrame([feature_row])

    # Keep exactly the same feature order used during training.
    X = X[feature_cols]

    prediction_id = model.predict(X)[0]

    fault = class_names[int(prediction_id)]

    probabilities = model.predict_proba(X)[0]

    confidence = probabilities[int(prediction_id)]

    location = FAULT_LOCATION.get(
        fault,
        "Unknown"
    )

    return {
        "fault": fault,
        "location": location,
        "confidence": confidence
    }


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    DATA_PATH = (
        "output/"
        "overlap_fault_features.csv"
    )

    df = pd.read_csv(DATA_PATH)

    print("\n=== NVMe FAULT LOCALIZATION ===")

    # One example from each class.
    example_indices = []

    for label in [
        "baseline",
        "cpu_contention",
        "network_congestion",
        "noisy_neighbour",
        "pcie_contention",
        "ssd_saturation"
    ]:

        matching = df.index[
            df["label"] == label
        ]

        if len(matching) > 0:
            example_indices.append(
                matching[0]
            )


    for idx in example_indices:

        row = df.loc[idx]

        result = localize_fault(row)

        print("\n" + "=" * 55)

        print(
            f"Predicted fault : "
            f"{result['fault']}"
        )

        print(
            f"Latency source  : "
            f"{result['location']}"
        )

        print(
            f"Confidence      : "
            f"{result['confidence']:.2%}"
        )

        # Used only to validate the demo.
        print(
            f"Actual label    : "
            f"{row['label']}"
        )