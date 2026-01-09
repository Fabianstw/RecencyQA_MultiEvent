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

DATASET_FILE = "C:\\ws2025\\aktuelle\\RecencyQA\\finetune\\test_eval.json"
OUTPUT_DIR = "results_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
   "schulzeschulbus_b2df/Qwen2.5-14B-Instruct-recency_QWEN2_5_14B-852a4208",
   "moonshotai/Kimi-K2-Instruct-0905",
   "Qwen/Qwen2.5-72B-Instruct-Turbo",
   "deepseek-ai/DeepSeek-V3"

    
]



TEMPERATURE = 0.0
MAX_TOKENS = 90

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
classify how often the data for the answer changes.

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

    def extract_label_from_text(raw):
        """Extrahiere das erste Label aus dem Text."""
        for label in LABELS:
            if label.lower() in raw.lower():
                return label
        return ""  # fallback, falls kein Label gefunden

    for sample in tqdm(dataset, desc=f"{model}"):
        pairs = extract_context_label_pairs(sample)

        for context, gold_label, ctx_id in pairs:
            prompt = build_prompt(sample["question"], context)

            # ===== API CALL =====
            try:
                if model.startswith("openai"):
                    # OpenAI GPT-OSS spezielle Behandlung
                    response = client.completions.create(
                        model=model,
                        prompt=prompt,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS
                    )
                    raw = response.choices[0].text.strip()
                    print(f"\n[DEBUG OPENAI RAW OUTPUT] q_id={sample['q_id']}, ctx={ctx_id}")
                    print(repr(raw))

                    # Label aus freiem Text extrahieren
                    pred = extract_label_from_text(raw)

                else:
                    # Standard JSON-Parsing für andere Modelle
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS
                    )
                    raw = response.choices[0].message.content.strip()
                    
                    pred = extract_label_from_text(raw)

            except Exception as e:
                print(f"API call failed for q_id={sample['q_id']}: {e}")
                pred = ""

            # Fallback falls kein Label gefunden wurde
            if not pred:
                pred = ""

            predictions.append({
                "q_id": sample["q_id"],
                "context_id": ctx_id,
                "question": sample["question"],
                "context": context,
                "gold_label": gold_label,
                "predicted_label": pred,
                "stationary": sample.get("stationary"),
                "event_dependency": sample.get("event_dependency"),
                "model": model
            })

    return predictions



def build_confusion_matrix(preds):
    n = len(LABELS)
    cm = [[0 for _ in range(n)] for _ in range(n)]

    for p in preds:
        g = p["gold_label"]
        pr = p["predicted_label"]

        if g in LABEL_TO_IDX and pr in LABEL_TO_IDX:
            gi = LABEL_TO_IDX[g]
            pi = LABEL_TO_IDX[pr]
            cm[gi][pi] += 1

    return cm

def per_label_stats(y_true, y_pred):
    stats = {}

    for i, label in enumerate(LABELS):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == i and yp == i)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != i and yp == i)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == i and yp != i)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        stats[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn
        }

    return stats


# =========================
# EVALUATION
# =========================

def evaluate(preds):
    y_true = []
    y_pred = []

    invalid = 0
    tolerant_correct = 0
    too_early = 0
    too_late = 0

    for p in preds:
        g = p["gold_label"]
        pr = p["predicted_label"]

        if g not in LABEL_TO_IDX or pr not in LABEL_TO_IDX:
            invalid += 1
            continue

        gi = LABEL_TO_IDX[g]
        pi = LABEL_TO_IDX[pr]

        y_true.append(gi)
        y_pred.append(pi)

        if abs(gi - pi) <= 1:
            tolerant_correct += 1

        if pi < gi:
            too_early += 1
        elif pi > gi:
            too_late += 1

    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)

    # ---- Metrics ----
    accuracy = correct / total if total else 0.0
    micro_f1 = accuracy  # multiclass property

    per_label = per_label_stats(y_true, y_pred)
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(LABELS)

    tolerant_f1 = tolerant_correct / total if total else 0.0

    confusion_matrix = build_confusion_matrix(preds)

    return {
        "count": total,
        "invalid": invalid,

        # Core metrics
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "tolerant_f1": tolerant_f1,

        # Error direction
        "error_direction": {
            "too_early": too_early,
            "too_late": too_late,
            "correct": correct
        },

        # Detailed analysis
        "per_label": per_label,
        "confusion_matrix": confusion_matrix
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
            groups[f"Stationary_{p.get('stationary')}"].append(p)
            groups[f"NumEvents_{p.get('num_events')}"].append(p)
            groups[f"Event_{p.get('event_dependency')}"].append(p)
            groups[f"GenType_{p.get('generation_type')}"].append(p)


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
                    f"MacroF1={v['macro_f1']:.3f}, "
                    f"MicroF1={v['micro_f1']:.3f}, "
                    f"TolF1={v['tolerant_f1']:.3f}, "
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