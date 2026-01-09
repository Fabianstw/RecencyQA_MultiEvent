import json
import os
from tqdm import tqdm
from together import Together
from collections import defaultdict

# =========================
# CONFIG
# =========================

DATASET_FILE = "C:\\ws2025\\aktuelle\\RecencyQA\\generatedSet\\recencyqa_betterMulti.json"
OUTPUT_DIR = "results_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    "openai/gpt-oss-120b"
]

TEMPERATURE = 0.0
MAX_TOKENS = 64  # erhöhte Tokens für JSON-Ausgabe

LABELS = [
    "An-Hour", "A-Few-Hours", "A-Day", "A-Few-Days",
    "A-Week", "A-Few-Weeks", "A-Month", "A-Few-Months",
    "A-Year", "A-Few-Years", "Many-Years", "Never"
]

LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

# =========================
# PROMPT-BUILDER
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

Return EXACTLY one label from:
{LABELS}

Return the answer as JSON in the format:
{{"label": "<ONE_OF_THE_LABELS>"}}

Do NOT include explanations or any additional text.
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
    pairs = []
    for idx, entry in enumerate(sample.get("labels", []), start=1):
        context_key = f"context{idx}"
        label_key = f"recency_label{idx}"
        if context_key in entry and label_key in entry:
            pairs.append((entry[context_key], entry[label_key], f"C{idx}"))
    return pairs

def run_inference(model, dataset, client):
    predictions = []

    for sample in tqdm(dataset, desc=f"{model}"):
        pairs = extract_context_label_pairs(sample)
        for context, gold_label, ctx_id in pairs:
            prompt = build_prompt(sample["question"], context)

            # ======= API CALL =======
            try:
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                raw = response.choices[0].text.strip()
            except Exception as e:
                print(f"API call failed for q_id={sample['q_id']}: {e}")
                raw = ""

            print(f"\n[DEBUG RAW OUTPUT] q_id={sample['q_id']}, ctx={ctx_id} ({len(raw)} chars): {repr(raw)}")

            # ======= JSON PARSING =======
            pred = ""
            if raw:
                try:
                    parsed = json.loads(raw)
                    pred = parsed.get("label", "")
                    if pred not in LABELS:
                        print(f"Warning: label '{pred}' not in LABELS")
                        pred = ""
                except Exception as e:
                    print(f"JSON parse error: {e} | raw={raw}")

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

# =========================
# MAIN
# =========================

def main():
    if "TOGETHER_API_KEY" not in os.environ:
        raise RuntimeError("Please set TOGETHER_API_KEY")

    client = Together()
    dataset = load_dataset(DATASET_FILE)

    for model in MODELS:
        preds = run_inference(model, dataset, client)
        out_file = os.path.join(OUTPUT_DIR, model.replace("/", "_") + "_predictions_debug.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\nPredictions saved to {out_file}")

if __name__ == "__main__":
    main()
