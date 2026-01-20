import json
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Load data
# ---------------------------
with open("summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Recency labels (must match dataset order)
labels = [
    "An-Hour", "A-Few-Hours", "A-Day", "A-Few-Days", "A-Week",
    "A-Few-Weeks", "A-Month", "A-Few-Months",
    "A-Year", "A-Few-Years", "Many-Years", "Never"
]

# ---------------------------
# Aggregate confusion matrices across models
# ---------------------------
agg = np.zeros((12, 12), dtype=float)

for entry in data:
    cm = np.array(entry["ALL"]["confusion_matrix"])
    agg += cm

# ---------------------------
# Row-normalize (recall view)
# ---------------------------
row_sums = agg.sum(axis=1, keepdims=True)
norm = np.divide(agg, row_sums, where=row_sums != 0)

# ---------------------------
# Plot heatmap
# ---------------------------
plt.figure(figsize=(9, 7))
im = plt.imshow(norm, aspect="auto")

plt.colorbar(im, fraction=0.046, pad=0.04, label="Fraction of samples")

plt.xticks(range(12), labels, rotation=45, ha="right")
plt.yticks(range(12), labels)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Model-averaged Recency Confusion Matrix (ALL)")

# Draw grid lines
for i in range(13):
    plt.axhline(i - 0.5, color="white", linewidth=0.5)
    plt.axvline(i - 0.5, color="white", linewidth=0.5)

plt.tight_layout()
plt.savefig("aggregated_confusion_heatmap.pdf")
plt.savefig("aggregated_confusion_heatmap.png", dpi=200)
plt.show()