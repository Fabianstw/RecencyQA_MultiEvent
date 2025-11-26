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
Generate EXACTLY 4 new temporal questions inspired by the examples.

Your questions MUST be a MIX of:
1) stable-update-frequency questions 
   (e.g., yearly updates, monthly reports, seasonal cycles)
2) variable-update-frequency questions 
   (e.g., crisis phases, weather conditions, market reactions, breaking events)

Rules:
- Must rely on information that CHANGES over time.
- Must include either a stable cyclical process OR an event-driven, rapidly shifting situation.
- Must be meaningful real-world questions.
- Must NOT paraphrase the examples.
- Must NOT use placeholders.

Example questions:
{examples}

Return VALID JSON ONLY in this EXACT structure:

{{
  "questions": [
    "question_1",
    "question_2",
    "question_3",
    "question_4"
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
Generate EXACTLY 4 new temporal questions that require comparing
or relating at least two different temporal events or phases.

Your questions MUST be a MIX of:
1) stable-update-frequency comparisons 
   (e.g., comparing two yearly events, two policy cycles)
2) variable-update-frequency comparisons 
   (e.g., comparing how two crises evolved across rapidly changing phases)

Rules:
- Each question must require temporal reasoning AND combine two different events.
- Must rely on information that changes over time.
- Must involve at least two distinct events.
- Must rely on information that CHANGES over time
- Must NOT paraphrase examples.
- Must NOT use placeholders.

Example questions:
{examples}

Return VALID JSON ONLY in this EXACT structure:

{{
  "questions": [
    "question_1",
    "question_2",
    "question_3",
    "question_4"
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

Your tasks (do them step by step internally, but only output the final JSON):

1. Provide a recency label (How frequently must the answer be updated to stay correct ?), (choose key from):
{{
  "An-Hour": "The answer changes within one hour",
  "A-Few-Hours": "The answer changes within a few hours",
  "A-Day": "The answer changes within one day",
  "A-Few-Days": "The answer changes within a few days",
  "A-Week": "The answer changes within one week",
  "A-Few-Weeks": "The answer changes within a few weeks",
  "A-Month": "The answer changes within one month",
  "A-Few-Months": "The answer changes within a few months",
  "A-Year": "The answer changes within one year",
  "A-Few-Years": "The answer changes within a few years",
  "Many-Years": "The answer will need 10 or more years to change",
  "Never": "The answer will never change"
}}

2. Provide a really short but clear temporal context in ONE sentence. 
    - The context should ground the question on an EVENT! 
    - Do not include specific years, simply create a natural moment where the question arises.
    - Only describe an EVENT, PHASE, or CONDITION that triggers the question, no “someone asks” or "someone wonders".

3. Determine whether the question is stationary or non-stationary.
Use this rule:
- "YES" (stationary) if:
  - The chosen recency label would remain the same regardless of when the question is asked.
  - Even if the factual answer changes regularly, the time frame of change remains consistent, so the label is stable.
- "NO" (non-stationary) only if:
  - The appropriate recency label itself would change depending on when the question is asked.
  - OR the question is only relevant within a short time window that makes its temporal behavior unstable.

Critical distinction: Questions mentioning speciﬁc events are stationary if they ask about metrics that change at the same frequency regardless of
the event.

4. If stationary is NO:
    4.1 Provide a second recency label (label2), following the same rules as in step 1.
    4.2 Provide a second context (context2), following the same rules as in step 2.


IMPORTANT:
Output VALID JSON ONLY.
ALWAYS output lists.
If stationary : lists have length 1.
If non-stationary : lists have length 2.
NEVER rename keys: must be exactly "recency_list", "context_list" and "stationary".

STRICT FORMAT:

{{
  "recency_list": ["label1", "label2 optional"],
  "context_list": ["context1", "context2 optional"],
  "stationary": "<YES or NO>"
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
        "recencyqa_plus_stationaryornot.json"
    )
