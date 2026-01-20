import json
from collections import Counter
import matplotlib.pyplot as plt
import squarify  # pip install squarify

# Load dataset
with open("recencyqa_OUR_DATASET.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect all recency labels
all_labels = []
for item in data:
    for lbl in item.get("labels", []):
        if "recency_label1" in lbl:
            all_labels.append(lbl["recency_label1"])
        if "recency_label2" in lbl:
            all_labels.append(lbl["recency_label2"])

# Count frequency
label_counts = Counter(all_labels)

# Prepare data for treemap
labels_text = [f"{k}\n({v})" for k, v in label_counts.items()]  # label + count
sizes = list(label_counts.values())
colors = plt.cm.tab20c.colors * ((len(sizes) // 20) + 1)

# Plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Get treemap rectangles
rects = squarify.normalize_sizes(sizes, 100, 100)
rects = squarify.squarify(rects, 0, 0, 100, 100)

# Draw each rectangle manually to control text and remove edges
for i, rect in enumerate(rects):
    x, y, dx, dy = rect['x'], rect['y'], rect['dx'], rect['dy']
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            dx,
            dy,
            facecolor=colors[i],
            edgecolor=None  # no border
        )
    )
    # Dynamic font size based on rectangle area
    area = dx * dy
    fontsize = max(8, min(int(area / 10), 20))  # min 8, max 20
    ax.text(
        x + dx/2,
        y + dy/2,
        labels_text[i],
        ha='center',
        va='center',
        fontsize=fontsize,
        wrap=True
    )

# Remove white space around the treemap
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save as SVG
plt.savefig("recency_labels_treemap.svg", format="svg", bbox_inches='tight', pad_inches=0)