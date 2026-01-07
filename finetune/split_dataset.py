"""
Prepare RecencyQA dataset for Together AI fine-tuning.

Features:
- Question-level stratified split (NO data leakage)
- Stratification over:
  event_dependency, num_events, stationary, generation_type
- Train / Dev / Test split (70 / 15 / 15)
- Flatten multi-context questions for Together AI
- Export Together-AI-compatible JSONL files
- Save test_eval.json in the original dataset structure

Author: (you)
"""

import json
import random
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# =========================
# CONFIG
# =========================

INPUT_JSON = "C:\\ws2025\\aktuelle\\RecencyQA\\finetune\\recencyqa_OUR_DATASET.json"  # your original dataset
OUT_TRAIN = "C:\\ws2025\\aktuelle\\RecencyQA\\finetune\\train.jsonl"
OUT_DEV   = "C:\\ws2025\\aktuelle\\RecencyQA\\finetune\\dev.jsonl"
OUT_TEST  = "C:\\ws2025\\aktuelle\\RecencyQA\\finetune\\test.jsonl"
OUT_EVAL  = "C:\\ws2025\\aktuelle\\RecencyQA\\finetune\\test_eval.json"

RANDOM_SEED = 42

# =========================
# LOAD DATASET
# =========================

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# STRATIFICATION
# =========================

def build_strat_label(sample):
    """
    Build a combined stratification label to preserve
    distribution across multiple dataset axes.
    """
    return "|".join([
        sample.get("event_dependency", "UNK"),
        str(sample.get("num_events", "UNK")),
        sample.get("stationary", "UNK"),
        sample.get("generation_type", "UNK")
    ])

def stratified_split(dataset, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    strat_labels = [build_strat_label(s) for s in dataset]

    # ---- Train vs Temp (70 / 30) ----
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed
    )
    train_idx, temp_idx = next(sss1.split(range(len(dataset)), strat_labels))

    train = [dataset[i] for i in train_idx]
    temp  = [dataset[i] for i in temp_idx]

    # ---- Dev vs Test (15 / 15) ----
    temp_labels = [build_strat_label(s) for s in temp]

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed
    )
    dev_idx, test_idx = next(sss2.split(range(len(temp)), temp_labels))

    dev  = [temp[i] for i in dev_idx]
    test = [temp[i] for i in test_idx]

    return train, dev, test

# =========================
# FLATTEN FOR TOGETHER AI
# =========================

def flatten_dataset(samples):
    """
    Convert question-level samples to
    question-context pairs for Together AI.
    """
    flat = []

    for s in samples:
        for entry in s["labels"]:
            for k, context in entry.items():
                if k.startswith("context"):
                    idx = k.replace("context", "")
                    label = entry.get(f"recency_label{idx}")

                    if not label:
                        continue

                    flat.append({
                        "q_id": s["q_id"],
                        "question": s["question"],
                        "context": context,
                        "gold_label": label,

                        # meta (kept for analysis/debugging)
                        "event_dependency": s.get("event_dependency"),
                        "num_events": s.get("num_events"),
                        "stationary": s.get("stationary"),
                        "generation_type": s.get("generation_type")
                    })

    return flat

# =========================
# TOGETHER JSONL EXPORT
# =========================

def to_together_jsonl(samples, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in temporal reasoning."
                    },
                    {
                        "role": "user",
                        "content": f"""Question:
{s['question']}

Context:
{s['context']}

Choose exactly one label from:
[An-Hour, A-Few-Hours, A-Day, A-Few-Days, A-Week, A-Few-Weeks,
 A-Month, A-Few-Months, A-Year, A-Few-Years, Many-Years, Never]

Answer ONLY with the label.""" 
                    },
                    {
                        "role": "assistant",
                        "content": s["gold_label"]
                    }
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# =========================
# SAVE JSON
# =========================

def save_eval_json(samples, out_path):
    """
    Save dataset in its original structure (unflattened)
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

# =========================
# STATS
# =========================

def print_split_stats(name, samples):
    labels = Counter(build_strat_label(s) for s in samples)
    print(f"\n{name} split ({len(samples)} questions)")
    for k, v in labels.most_common(10):
        print(f"  {k}: {v}")

# =========================
# MAIN
# =========================

def main():
    dataset = load_dataset(INPUT_JSON)

    train_q, dev_q, test_q = stratified_split(dataset, RANDOM_SEED)

    print(f"\nQuestion counts:")
    print(f"  Train: {len(train_q)}")
    print(f"  Dev:   {len(dev_q)}")
    print(f"  Test:  {len(test_q)}")

    print_split_stats("Train", train_q)
    print_split_stats("Dev", dev_q)
    print_split_stats("Test", test_q)

    # Flatten Train / Dev / Test für Together AI
    train_flat = flatten_dataset(train_q)
    dev_flat   = flatten_dataset(dev_q)
    test_flat  = flatten_dataset(test_q)

    print(f"\nInstance counts (after flattening):")
    print(f"  Train: {len(train_flat)}")
    print(f"  Dev:   {len(dev_flat)}")
    print(f"  Test:  {len(test_flat)}")

    # Together AI JSONL Export
    to_together_jsonl(train_flat, OUT_TRAIN)
    to_together_jsonl(dev_flat,   OUT_DEV)
    to_together_jsonl(test_flat,  OUT_TEST)

    # Test-Eval JSON: gleiche Struktur wie Input-Dataset
    save_eval_json(test_q, OUT_EVAL)

    print("\nDone.")
    print(f"Files written:")
    print(f"  {OUT_TRAIN}")
    print(f"  {OUT_DEV}")
    print(f"  {OUT_TEST}")
    print(f"  {OUT_EVAL}")

if __name__ == "__main__":
    main()
