"""
Recency classification experiment with:
- Context-aware prompting
- Stationary vs Non-stationary comparison
- Single-Event vs Multi-Event comparison
- Support for variable number of contexts per question
"""

import json
import os
from tqdm import tqdm
from together import Together
from collections import defaultdict

# =========================
# CONFIG
# =========================

DATASET_FILE = "C:\\ws2025\\aktuelle\\RecencyQA\\generatedSet\\recencyqa_betterMulti.json"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-R1",
    "moonshotai_Kimi-K2-Instruct-0905"
]

TEMPERATURE = 0.0
MAX_TOKENS = 10

LABELS = [
    "An-Hour", "A-Few-Hours", "A-Day", "A-Few-Days",
    "A-Week", "A-Few-Weeks", "A-Month", "A-Few-Months",
    "A-Year", "A-Few-Years", "Many-Years", "Never"
]

LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

# =========================
# PROMPT
# =========================

def build_prompt(question: str, context: str) -> str:
    return f"""
You are an expert in temporal reasoning.

Given the following question and its temporal context,
classify the needed recency of the data for the answer.

Question:
{question}

Context:
{context}

Choose exactly one label from:
[An-Hour, A-Few-Hours, A-Day, A-Few-Days, A-Week, A-Few-Weeks,
 A-Month, A-Few-Months, A-Year, A-Few-Years, Many-Years, Never]

Answer ONLY with the label.
DO NOT provide any explanations or additional text.
""".strip()

# =========================
# DATA LOADING
# =========================

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# INFERENCE
# =========================

def extract_context_label_pairs(sample):
    """
    Converts the 'labels' list into (context, gold_label, context_id) tuples.
    """
    pairs = []

    for idx, entry in enumerate(sample["labels"], start=1):
        context_key = f"context{idx}"
        label_key = f"recency_label{idx}"

        if context_key in entry and label_key in entry:
            pairs.append(
                (entry[context_key], entry[label_key], f"C{idx}")
            )

    return pairs

def run_inference(model, dataset, client):
    predictions = []

    for sample in tqdm(dataset, desc=f"{model}"):
        pairs = extract_context_label_pairs(sample)

        for context, gold_label, ctx_id in pairs:
            prompt = build_prompt(sample["question"], context)

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            pred = response.choices[0].message.content.strip()

            predictions.append({
                "q_id": sample["q_id"],
                "context_id": ctx_id,
                "question": sample["question"],
                "context": context,
                "gold_label": gold_label,
                "predicted_label": pred,
                "stationary": sample["stationary"],
                "event_dependency": sample["event_dependency"],
                "model": model
            })

    return predictions

# =========================
# EVALUATION
# =========================

def evaluate(preds):
    total = correct = tolerant = invalid = 0

    for p in preds:
        g = p["gold_label"]
        pr = p["predicted_label"]

        if g not in LABEL_TO_IDX or pr not in LABEL_TO_IDX:
            invalid += 1
            continue

        gi = LABEL_TO_IDX[g]
        pi = LABEL_TO_IDX[pr]

        total += 1
        if gi == pi:
            correct += 1
        if abs(gi - pi) <= 1:
            tolerant += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "tolerant_accuracy": tolerant / total if total else 0.0,
        "count": total,
        "invalid": invalid
    }

# =========================
# MAIN
# =========================

def main():
    if "TOGETHER_API_KEY" not in os.environ:
        raise RuntimeError("Please set TOGETHER_API_KEY")

    client = Together()
    dataset = load_dataset(DATASET_FILE)

    final_summary = []

    for model in MODELS:
        preds = run_inference(model, dataset, client)

        out_file = os.path.join(
            OUTPUT_DIR,
            model.replace("/", "_") + "_predictions.jsonl"
        )

        with open(out_file, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

        # Grouped evaluation
        groups = defaultdict(list)

        for p in preds:
            groups["ALL"].append(p)
            groups[f"Stationary_{p['stationary']}"].append(p)
            groups[f"Event_{p['event_dependency']}"].append(p)

        model_result = {"model": model}

        for gname, gpreds in groups.items():
            model_result[gname] = evaluate(gpreds)

        final_summary.append(model_result)

    # Save summary
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    # Print summary
    print("\n=== FINAL RESULTS ===")
    for m in final_summary:
        print(f"\nModel: {m['model']}")
        for k, v in m.items():
            if k != "model":
                print(
                    f"{k}: "
                    f"Acc={v['accuracy']:.3f}, "
                    f"TolAcc={v['tolerant_accuracy']:.3f}, "
                    f"N={v['count']}"
                )

if __name__ == "__main__":
    main()


# confusion matrix, per-label stats, etc. can be added here
#error_type  analysis
# f1 
# precision
# recall

# maybe ask for reasoning  of the model