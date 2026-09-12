import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "output/final_feature_importance.csv"
OUTPUT_FILE = "output/final_feature_importance.png"

# --------------------------------------------------
# Load feature importance
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# Take top 20
top_features = (
    df.sort_values("importance", ascending=False)
      .head(20)
      .sort_values("importance")
)

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 8))

ax.barh(
    top_features["feature"],
    top_features["importance"]
)

ax.set_title(
    "Top 20 Features for NVMe Fault Localization"
)

ax.set_xlabel("Random Forest Feature Importance")
ax.set_ylabel("Telemetry Feature")

ax.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {OUTPUT_FILE}")