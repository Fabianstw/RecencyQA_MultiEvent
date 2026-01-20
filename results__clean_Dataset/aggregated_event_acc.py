import json
from collections import defaultdict

# -----------------------
# Recency label order
# -----------------------
LABEL_ORDER = [
    "An-Hour",
    "A-Few-Hours",
    "A-Day",
    "A-Few-Days",
    "A-Week",
    "A-Few-Weeks",
    "A-Month",
    "A-Few-Months",
    "A-Year",
    "A-Few-Years",
    "Many-Years",
    "Never",
]

label_to_idx = {label: i for i, label in enumerate(LABEL_ORDER)}

# -----------------------
# File paths
# -----------------------
PREDICTION_FILES = [
    "Qwen_Qwen2.5-72B-Instruct-Turbo_predictions.jsonl",
    "moonshotai_Kimi-K2-Instruct-0905_predictions.jsonl",
    "deepseek-ai_DeepSeek-V3_predictions.jsonl",
]

DATASET_FILE = "../NewDataset/recencyqa_OUR_DATASET.json"


# -----------------------
# Load dataset metadata
# -----------------------
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

def tolerant_correct(gold, pred, tolerance=1):
  if gold not in label_to_idx or pred not in label_to_idx:
      return False

  return abs(label_to_idx[gold] - label_to_idx[pred]) <= tolerance

# q_id -> metadata
qid_meta = {}

for item in dataset:
    qid_meta[item["q_id"]] = {
        "num_events": item.get("num_events"),
        "event_dependency": item.get("event_dependency"),
        # generation_type may not exist for older questions
        "generation_type": item.get("generation_type"),
    }

# -----------------------
# Accuracy counters
# -----------------------
# stats[model][num_events]
stats = defaultdict(lambda: defaultdict(lambda: {
    "correct": 0,
    "tolerant_correct": 0,
    "total": 0
}))

agg_stats = defaultdict(lambda: {
    "correct": 0,
    "tolerant_correct": 0,
    "total": 0
})

# Multi-event causal / temporal
multi_stats = defaultdict(lambda: defaultdict(lambda: {
    "correct": 0,
    "tolerant_correct": 0,
    "total": 0
}))

agg_multi_stats = defaultdict(lambda: {
    "correct": 0,
    "tolerant_correct": 0,
    "total": 0
})

# -----------------------
# Process prediction files
# -----------------------
for filepath in PREDICTION_FILES:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            q_id = row["q_id"]
            model = row["model"]
            gold = row["gold_label"]
            pred = row["predicted_label"]

            if q_id not in qid_meta:
                continue

            meta = qid_meta[q_id]
            num_events = meta["num_events"]
            event_dep = meta["event_dependency"]
            gen_type = meta["generation_type"]

            exact_correct = gold == pred
            tol_correct = tolerant_correct(gold, pred, tolerance=1)

            # -----------------------
            # Per num_events
            # -----------------------
            stats[model][num_events]["total"] += 1
            if exact_correct:
                stats[model][num_events]["correct"] += 1
            if tol_correct:
                stats[model][num_events]["tolerant_correct"] += 1

            agg_stats[num_events]["total"] += 1
            if exact_correct:
                agg_stats[num_events]["correct"] += 1
            if tol_correct:
                agg_stats[num_events]["tolerant_correct"] += 1

            # -----------------------
            # Multi-event only: causal vs temporal
            # -----------------------
            if event_dep == "Multi-Event" and gen_type in {"causal", "temporal_only"}:
              multi_stats[model][gen_type]["total"] += 1
              if exact_correct:
                  multi_stats[model][gen_type]["correct"] += 1
              if tol_correct:
                  multi_stats[model][gen_type]["tolerant_correct"] += 1

              agg_multi_stats[gen_type]["total"] += 1
              if exact_correct:
                  agg_multi_stats[gen_type]["correct"] += 1
              if tol_correct:
                  agg_multi_stats[gen_type]["tolerant_correct"] += 1

# -----------------------
# Print results
# -----------------------
print("\nAccuracy by model and number of events:\n")

for model, model_stats in stats.items():
    print(f"Model: {model}")
    for num_events in sorted(model_stats.keys()):
        c = model_stats[num_events]["correct"]
        t = model_stats[num_events]["total"]
        acc = c / t if t else 0.0

        label = (
            "Single-Event" if num_events == 1
            else "Two-Event" if num_events == 2
            else "Three-Event"
        )

        print(f"  {label}: {acc:.3f} ({c}/{t})")
    print()

print("Aggregated accuracy across all models (by number of events):\n")
for num_events in sorted(agg_stats.keys()):
    c = agg_stats[num_events]["correct"]
    t = agg_stats[num_events]["total"]
    acc = c / t if t else 0.0

    label = (
        "Single-Event" if num_events == 1
        else "Two-Event" if num_events == 2
        else "Three-Event"
    )

    print(f"  {label}: {acc:.3f} ({c}/{t})")

# -----------------------
# Multi-event causal vs temporal
# -----------------------
print("\nMulti-Event accuracy by model (causal vs. temporal_only):\n")

for model, gen_stats in multi_stats.items():
    print(f"Model: {model}")
    for gen_type in ["causal", "temporal_only"]:
        c = gen_stats[gen_type]["correct"]
        t = gen_stats[gen_type]["total"]
        acc = c / t if t else 0.0

        print(f"  {gen_type}: {acc:.3f} ({c}/{t})")
    print()

print("Aggregated Multi-Event accuracy across all models:\n")
for gen_type in ["causal", "temporal_only"]:
    c = agg_multi_stats[gen_type]["correct"]
    t = agg_multi_stats[gen_type]["total"]
    acc = c / t if t else 0.0

    print(f"  {gen_type}: {acc:.3f} ({c}/{t})")


print("\nAccuracy by model and number of events:\n")

for model, model_stats in stats.items():
    print(f"Model: {model}")
    for num_events in sorted(model_stats.keys()):
        c = model_stats[num_events]["correct"]
        tc = model_stats[num_events]["tolerant_correct"]
        t = model_stats[num_events]["total"]

        acc = c / t if t else 0.0
        tol_acc = tc / t if t else 0.0

        label = (
            "Single-Event" if num_events == 1
            else "Two-Event" if num_events == 2
            else "Three-Event"
        )

        print(
            f"  {label}: "
            f"Exact={acc:.3f}, "
            f"±1={tol_acc:.3f} "
            f"({c}/{t}, tol {tc}/{t})"
        )
    print()

print("Aggregated accuracy across all models (by number of events):\n")
for num_events in sorted(agg_stats.keys()):
    c = agg_stats[num_events]["correct"]
    tc = agg_stats[num_events]["tolerant_correct"]
    t = agg_stats[num_events]["total"]

    acc = c / t if t else 0.0
    tol_acc = tc / t if t else 0.0

    label = (
        "Single-Event" if num_events == 1
        else "Two-Event" if num_events == 2
        else "Three-Event"
    )

    print(
        f"  {label}: "
        f"Exact={acc:.3f}, "
        f"±1={tol_acc:.3f} "
        f"({c}/{t}, tol {tc}/{t})"
    )