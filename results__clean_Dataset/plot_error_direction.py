import json
import matplotlib.pyplot as plt
import numpy as np

MODEL_LABELS = {
    "schulzeschulbus_b2df/Qwen2.5-14B-Instruct-recency_QWEN2_5_14B-852a4208": "Qwen2.5-14B (FT)",
    "moonshotai/Kimi-K2-Instruct-0905": "Kimi",
    "Qwen/Qwen2.5-72B-Instruct-Turbo": "Qwen2.5-72B",
    "deepseek-ai/DeepSeek-V3": "DeepSeek-V3",
}

# ---------------------------
# Load summary.json
# ---------------------------
with open("summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------------------------
# Extract error direction counts (ALL split)
# ---------------------------
models = []
too_early = []
too_late = []

for entry in data:
    model_name = entry["model"]
    err = entry["ALL"]["error_direction"]

    models.append(MODEL_LABELS.get(model_name, model_name.split("/")[-1]))
    too_early.append(err["too_early"])
    too_late.append(err["too_late"])

# ---------------------------
# Plot
# ---------------------------
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, too_early, width, label="Too early")
plt.bar(x + width/2, too_late,  width, label="Too late")

plt.xticks(x, models, rotation=15)
plt.ylabel("Number of predictions")
plt.title("Directional Recency Errors per Model")
plt.legend()

# Annotate bars
for i in range(len(models)):
    plt.text(x[i] - width/2, too_early[i] + 10, str(too_early[i]),
             ha="center", va="bottom", fontsize=9)
    plt.text(x[i] + width/2, too_late[i] + 10, str(too_late[i]),
             ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("error_direction_barplot.pdf")
plt.savefig("error_direction_barplot.png", dpi=200)
plt.show()