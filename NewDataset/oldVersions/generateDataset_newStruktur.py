import json
import pandas as pd
from tqdm import tqdm
from together import Together


########################################################
# 1. Together AI Setup
########################################################

client = Together()
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
print("Using Together AI model:", MODEL_NAME)


########################################################
# 2. Chat wrapper with token logging + JSON extraction
########################################################

def llm(prompt, max_new_tokens=350):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.95
    )

    # Token logging
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    print(f"[Token Log] Input={input_tokens} | Output={output_tokens} | Total={usage.total_tokens}")

    cost = input_tokens * 0.27/1e6 + output_tokens * 0.54/1e6
    print(f"[Estimated cost] ${cost:.6f}\n")

    return response.choices[0].message.content


def extract_json(text):
    """Extract JSON object from LLM output safely."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except:
        return None


########################################################
# 3. JSON/JSONL auto loader
########################################################

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)

    if first == "[":
        print("Detected JSON → array")
        return pd.read_json(path)
    return pd.read_json(path, lines=True)


########################################################
# 4. QUESTION GENERATION (stable)
########################################################

SINGLE_GEN_PROMPT = """
You are a temporal question generation system.

Your task:
Generate EXACTLY 2 new temporal questions inspired by the examples.
Rules:
- Must require temporal reasoning.
- Must be meaningful real-world questions.
- Must NOT paraphrase the examples.
- Must NOT use placeholders.

Example questions:
{examples}

Return VALID JSON ONLY in this EXACT structure:

{{
  "questions": [
    "question_1",
    "question_2"
  ]
}}
"""

def generate_single_questions(example_questions):
    ex = "\n".join(f"- {q}" for q in example_questions)
    result = llm(SINGLE_GEN_PROMPT.format(examples=ex))
    js = extract_json(result)
    return js["questions"] if js and "questions" in js else []


MULTI_GEN_PROMPT = """
You are a multi-event temporal reasoning question generator.

Your task:
Generate EXACTLY 2 new temporal questions requiring multiple temporal events/phases.
Rules:
- Must involve at least two distinct events.
- Must require temporal reasoning.
- Must NOT paraphrase examples.
- Must NOT use placeholders.

Example questions:
{examples}

Return VALID JSON ONLY in this EXACT structure:

{{
  "questions": [
    "question_1",
    "question_2"
  ]
}}
"""

def generate_multi_questions(example_questions):
    ex = "\n".join(f"- {q}" for q in example_questions)
    result = llm(MULTI_GEN_PROMPT.format(examples=ex))
    js = extract_json(result)
    return js["questions"] if js and "questions" in js else []


########################################################
# 5. LABELING (model decides 1 or 2 contexts)
########################################################

LABEL_PROMPT = """
Analyze this question and produce temporal labels:

"{question}"

Your tasks:

1. Decide whether the question naturally has ONE or TWO meaningful temporal contexts.
2. For each context:
   - Provide a recency label (How frequently must the answer be updated to stay correct ?), (choose from):
     ["An-Hour","A-Few-Hours","A-Day","A-Few-Days","A-Week","A-Few-Weeks",
      "A-Month","A-Few-Months","A-Year","A-Few-Years","Many-Years","Never"]
   - Provide a short but clear temporal context description in one sentence, describing, when this question is asked and why this label makes sense. The context
    should ground the question in an event. Do not include specific years simply create a moment where the question arises and the label is appropriate.
     

IMPORTANT:
- Output VALID JSON ONLY.
- ALWAYS output lists.
- If ONE context: lists have length 1.
- If TWO contexts: lists have length 2.
- NEVER rename keys: must be exactly "recency_list" and "context_list".

STRICT FORMAT:

{{
  "recency_list": ["label1", "label2 optional"],
  "context_list": ["context1", "context2 optional"]
}}
"""

def label_question(q):
    result = llm(LABEL_PROMPT.format(question=q))
    return extract_json(result)


########################################################
# 6. RECONSTRUCT FINAL FORMAT
########################################################

def build_final_labels(raw_labels):
    """Convert list-format model output → your desired key structure."""
    final_list = []

    recs = raw_labels["recency_list"]
    ctxs = raw_labels["context_list"]

    for i, (r, c) in enumerate(zip(recs, ctxs), start=1):
        final_list.append({
            f"recency_label{i}": r,
            f"context{i}": c
        })

    if len(recs) == 1:
        stationary_flag = "YES"
    else:
        stationary_flag = "NO"

    return final_list, stationary_flag



########################################################
# 7. MAIN PIPELINE
########################################################

def generate_recencyqa_plus(input_path, output_path):

    df = load_dataset(input_path)
    output = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        example_questions = [row["question"]]

        # Generate questions
        qs_single = generate_single_questions(example_questions)
        qs_multi = generate_multi_questions(example_questions)

        # --- SINGLE EVENT QUESTIONS ---
        for q in qs_single:
            raw = label_question(q)
            if not raw:
                continue

            labels, stationary = build_final_labels(raw)

            output.append({
                "q_id": row["q_id"],
                "question": q,
                "event_dependency": "Single-Event",
                "labels": labels,
                "stationary": stationary
            })

        # --- MULTI EVENT QUESTIONS ---
        for q in qs_multi:
            raw = label_question(q)
            if not raw:
                continue

            labels, stationary = build_final_labels(raw)

            output.append({
                "q_id": row["q_id"],
                "question": q,
                "event_dependency": "Multi-Event",
                "labels": labels,
                "stationary": stationary
            })

    # Save to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for o in output:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print("Saved:", output_path)


###############################################
# RUN
###############################################

if __name__ == "__main__":
    generate_recencyqa_plus(
        "E:\\Uni\\ws2025\\aktuelleThemen\\RecencyQA\\NewDataset\\RecencyQA_dataset_small.json",
        "recencyqa_plus_FINAL.json"
    )
