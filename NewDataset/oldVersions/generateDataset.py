import json
import pandas as pd
from tqdm import tqdm
from together import Together


########################################################
# 1. Together AI Setup
########################################################

# Make sure your environment variable TOGETHER_API_KEY is set!
client = Together()

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

print("Using Together AI model:", MODEL_NAME)


########################################################
# 2. Chat wrapper (replaces local HF model)
########################################################

def llm(prompt, max_new_tokens=350):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.95
    )

    # --- Token Logging ---
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    print(f"[Token Log] Input: {input_tokens} | Output: {output_tokens} | Total: {total_tokens}")

    # Optional: Kosten schätzen (Together AI Preisstand 2025)
    # Llama 3.3 70B Turbo
    # input:  $0.27 / 1M tokens
    # output: $0.54 / 1M tokens

    cost_input = input_tokens * 0.27 / 1_000_000
    cost_output = output_tokens * 0.54 / 1_000_000
    cost_total = cost_input + cost_output

    print(f"[Cost Est.] ${cost_total:.6f} USD\n")

    return response.choices[0].message.content



########################################################
# JSON/JSONL AUTO-DETECT LOADER
########################################################

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)

    if first == "[":
        print("Detected JSON array → using lines=False")
        return pd.read_json(path)
    elif first == "{":
        print("Detected JSONL → using lines=True")
        return pd.read_json(path, lines=True)
    else:
        raise ValueError("Invalid JSON/JSONL format")


########################################################
# 3. New Question Generation (Single)
########################################################

SINGLE_GEN_PROMPT = """
You are a question generation system.

Generate 5 NEW temporal questions inspired by the following examples.
Rules:
- DO NOT paraphrase the example questions.
- DO NOT use placeholders ("q1", "question 1", etc.).
- New questions MUST require time-based or evolving information.
- Questions MUST be meaningful real-world questions.

Example questions:
{examples}

Return JSON ONLY:
{{
 "questions": [
   "full natural question 1",
   "full natural question 2",
   "full natural question 3",
   "full natural question 4",
   "full natural question 5"
 ]
}}
"""

def generate_single_questions(example_questions):
    ex = "\n".join(f"- {q}" for q in example_questions)
    out = llm(SINGLE_GEN_PROMPT.format(examples=ex))

    try:
        j = json.loads(out[out.index("{"):out.rindex("}")+1])
        return j["questions"]
    except:
        return []


########################################################
# 4. Multi-event generation
########################################################

MULTI_GEN_PROMPT = """
Generate 5 multi-event temporal reasoning questions inspired by the following examples.

Rules:
- Each question MUST depend on at least two different situations/events.
- Each question MUST require temporal reasoning (before, after, since, while).
- DO NOT paraphrase the examples.
- NO placeholders ("q1", "example 1", etc.).
- Questions MUST be meaningful and realistic.

Example questions:
{examples}

Return JSON ONLY:
{{
 "questions": [
   "complex temporal question 1",
   "complex temporal question 2",
   "complex temporal question 3",
   "complex temporal question 4",
   "complex temporal question 5"
 ]
}}
"""

def generate_multi_questions(example_questions):
    ex = "\n".join(f"- {q}" for q in example_questions)
    out = llm(MULTI_GEN_PROMPT.format(examples=ex))

    try:
        j = json.loads(out[out.index("{"):out.rindex("}")+1])
        return j["questions"]
    except:
        return []


########################################################
# 5. Labeling
########################################################

LABEL_PROMPT = """
Label the following question:

"{question}"

Rules:
- NEVER output placeholders like "..." or "label".
- recency MUST be one of:
["An-Hour","A-Few-Hours","A-Day","A-Few-Days","A-Week","A-Few-Weeks","A-Month","A-Few-Months","A-Year",
 "A-Few-Years","Many-Years","Never"]
- stationarity MUST be "Stationary" or "Non-Stationary"
- event_dependency MUST be "Single-Event" or "Multi-Event"

Return JSON ONLY:
{{
 "recency": "<label>",
 "stationarity": "<label>",
 "event_dependency": "<label>",
 "context": "short context"
}}
"""

def label_question(q, is_multi):
    prompt = LABEL_PROMPT.format(question=q)
    if is_multi:
        prompt += "\n(event_dependency MUST be Multi-Event)\n"

    out = llm(prompt)

    try:
        j = json.loads(out[out.index("{"):out.rindex("}")+1])
        return j
    except:
        return None


########################################################
# 6. MAIN PIPELINE
########################################################

def generate_recencyqa_plus(input_path, output_path):

    df = load_dataset(input_path)
    output = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        example_questions = [row["question"]]

        qs1 = generate_single_questions(example_questions)
        qs2 = generate_multi_questions(example_questions)

        for q in qs1:
            labels = label_question(q, is_multi=False)
            if labels:
                output.append({
                    "source_qid": row["q_id"],
                    "generated_type": "single",
                    "question": q,
                    "labels": labels
                })

        for q in qs2:
            labels = label_question(q, is_multi=True)
            if labels:
                output.append({
                    "source_qid": row["q_id"],
                    "generated_type": "multi",
                    "question": q,
                    "labels": labels
                })

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
        "recencyqa_plus.json"
    )
