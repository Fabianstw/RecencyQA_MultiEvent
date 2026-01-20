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
# 2. Chat wrapper + JSON extraction
########################################################

def llm(prompt, max_new_tokens=350):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
    )

    usage = response.usage
    print(f"[Token Log] Input={usage.prompt_tokens} | Output={usage.completion_tokens}")

    cost = usage.prompt_tokens * 0.27/1e6 + usage.completion_tokens * 0.54/1e6
    print(f"[Estimated cost] ${cost:.6f}\n")

    return response.choices[0].message.content


def extract_json(text):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


########################################################
# 3. Dataset loader
########################################################

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)

    if first == "[":
        print("Detected JSON array")
        return pd.read_json(path)
    return pd.read_json(path, lines=True)


########################################################
# 4. GENERATION PROMPTS (4 pipelines)
########################################################

### --- Stationary + Single Event ---

PROMPT_STATIONARY_SINGLE = """
You generate STATIONARY temporal questions.

Definition (internal):
- A stationary question has a stable temporal update behavior.
- The answer changes over time, but the timespan how often the answer must be updated
  would remain the same regardless of when the question is asked.

Task:
Generate EXACTLY 2 stationary temporal questions focusing on ONE event or process.

Rules:
- Must rely on real-world phenomena that change over time.
- Must be stable, cyclical, predictable, or rhythm-based.
- Must NOT paraphrase the examples.
- Must NOT use placeholders.
- Do NOT mention stationarity in the question.

Examples:
{examples}

Return JSON:
{{
  "questions": ["q1","q2"]
}}
"""

def gen_stationary_single(example_q):
    ex = "\n".join(f"- {q}" for q in example_q)
    js = extract_json(llm(PROMPT_STATIONARY_SINGLE.format(examples=ex)))
    return js["questions"] if js else []


### --- Stationary + Multi Event ---

PROMPT_STATIONARY_MULTI = """
You generate STATIONARY multi-event temporal questions.

Task:
Generate EXACTLY 2 stationary temporal questions that involve TWO events or phases.
Generate ONE of those questions with events that are related causally. The other question must be generated with events from clearly different real-world domains.

Rules:
- Must involve at least TWO distinct temporal events.
- Temporal behavior must be stable, predictable, cyclical, or regular.
- Must rely on real-world change.
- Do NOT paraphrase examples.
- Do NOT use placeholders.
- Do NOT mention stationarity.

Examples:
{examples}

Return JSON:
{{
  "questions": ["q1","q2"]
}}
"""

def gen_stationary_multi(example_q):
    ex = "\n".join(f"- {q}" for q in example_q)
    js = extract_json(llm(PROMPT_STATIONARY_MULTI.format(examples=ex)))
    return js["questions"] if js else []


### --- Non-Stationary + Single Event ---

PROMPT_NONSTATIONARY_SINGLE = """
You generate NON-STATIONARY temporal questions.

Definition (internal):
- A non-stationary question has unstable temporal update behavior.
- How frequently the answer must be updated depends strongly on WHEN the question is asked.
- OR the question is only relevant within short, event-dependent windows.

Task:
Generate EXACTLY 2 non-stationary temporal questions focusing on ONE event.

Rules:
- Must rely on quickly evolving or unstable processes.
- Must be meaningful and real-world.
- Do NOT paraphrase examples.
- Do NOT use placeholders.
- Do NOT mention non-stationarity explicitly.

Examples:
{examples}

Return JSON:
{{
  "questions": ["q1","q2"]
}}
"""

def gen_nonstationary_single(example_q):
    ex = "\n".join(f"- {q}" for q in example_q)
    js = extract_json(llm(PROMPT_NONSTATIONARY_SINGLE.format(examples=ex)))
    return js["questions"] if js else []


### --- Non-Stationary + Multi Event ---

PROMPT_NONSTATIONARY_MULTI = """
You generate NON-STATIONARY multi-event temporal questions.

Task:
Generate EXACTLY 2 non-stationary temporal questions involving at least TWO events.
Generate ONE of those questions with events that are related causally. The other question must be generated with events from clearly different real-world domains.
Rules:
- At least one event must be unstable, unpredictable, or highly time-sensitive.
- Must rely on real-world temporal change.
- Must involve at least TWO distinct temporal events.
- Do NOT paraphrase examples.
- Do NOT use placeholders.
- Do NOT mention non-stationarity.

Examples:
{examples}

Return JSON:
{{
  "questions": ["q1","q2"]
}}
"""

def gen_nonstationary_multi(example_q):
    ex = "\n".join(f"- {q}" for q in example_q)
    js = extract_json(llm(PROMPT_NONSTATIONARY_MULTI.format(examples=ex)))
    return js["questions"] if js else []


########################################################
# 5. LABEL PROMPTS (NO stationarity inside prompt)
########################################################

LABEL_STATIONARY_PROMPT = """
Analyze this temporal question and produce temporal labels:

"{question}"

Your tasks (internal reasoning only, output JSON only):

1. Provide ONE recency label:
{{
  "An-Hour": "changes within one hour",
  "A-Few-Hours": "changes within a few hours",
  "A-Day": "changes within one day",
  "A-Few-Days": "changes within a few days",
  "A-Week": "changes within one week",
  "A-Few-Weeks": "changes within a few weeks",
  "A-Month": "changes within one month",
  "A-Few-Months": "changes within a few months",
  "A-Year": "changes within one year",
  "A-Few-Years": "changes within a few years",
  "Many-Years": "changes after 10 or more years",
  "Never": "never changes"
}}

2. Based on the selected label, write a short temporal context (ONE sentence) describing the current event, phase, or condition in which the question is asked.
   - Must describe an EVENT, PHASE or CONDITION, taking place when the question becomes relevant.
   - No specific years.
   - No meta reasoning.
   - No "current"
   - No "the question is asked"

Return JSON:
{{
  "recency_list": ["label1"],
  "context_list": ["context1"]
}}
"""

def label_stationary(q):
    js = extract_json(llm(LABEL_STATIONARY_PROMPT.format(question=q)))
    return js


LABEL_NONSTATIONARY_PROMPT = """
Analyze this temporal question and produce temporal labels:

"{question}"

Your tasks (internal reasoning only, output JSON only):

1. Provide TWO recency labels (for two different realistic temporal situations), choose from:
{{
  "An-Hour": "changes within one hour",
  "A-Few-Hours": "changes within a few hours",
  "A-Day": "changes within one day",
  "A-Few-Days": "changes within a few days",
  "A-Week": "changes within one week",
  "A-Few-Weeks": "changes within a few weeks",
  "A-Month": "changes within one month",
  "A-Few-Months": "changes within a few months",
  "A-Year": "changes within one year",
  "A-Few-Years": "changes within a few years",
  "Many-Years": "changes after 10 or more years",
  "Never": "never changes"
}}

2. For each selected label provide ONE short temporal contexts (ONE sentence each) describing the current event, phase, or condition in which the question is asked.
   - Each context must describe a different  EVENT, PHASE or CONDITION, taking place when the question becomes relevant.
   - No specific years.
   - No meta reasoning.
   - No "current"
   - No "the question is asked"

Return JSON:
{{
  "recency_list": ["label1", "label2"],
  "context_list": ["context1", "context2"]
}}
"""

def label_nonstationary(q):
    js = extract_json(llm(LABEL_NONSTATIONARY_PROMPT.format(question=q)))
    return js


########################################################
# 6. Label list conversion
########################################################

def build_final_labels(raw):
    final = []
    recs = raw.get("recency_list", [])
    ctxs = raw.get("context_list", [])

    for i, (r, c) in enumerate(zip(recs, ctxs), start=1):
        final.append({
            f"recency_label{i}": r,
            f"context{i}": c
        })

    return final


########################################################
# 7. Helper: ensure exact N
########################################################

def ensure_n(gen_func, example_q, n):
    out = []
    while len(out) < n:
        qs = gen_func(example_q)
        out.extend(qs)
        if not qs:
            break
    return out[:n]


########################################################
# 8. MAIN PIPELINE
########################################################

def generate_recencyqa_4way(
    input_path,
    output_path,
    n_single_stationary=2,
    n_multi_stationary=2,
    n_single_nonstat=2,
    n_multi_nonstat=2
):

    df = load_dataset(input_path)
    output = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        example = [row["question"]]

        # ---- generate 4 category sets ----
        qs_ss = ensure_n(gen_stationary_single, example, n_single_stationary)
        qs_sm = ensure_n(gen_stationary_multi, example, n_multi_stationary)
        qs_ns = ensure_n(gen_nonstationary_single, example, n_single_nonstat)
        qs_nm = ensure_n(gen_nonstationary_multi, example, n_multi_nonstat)

        # ---- STATIONARY SINGLE ----
        for q in qs_ss:
            raw = label_stationary(q)
            if raw:
                output.append({
                    "q_id": row["q_id"],
                    "question": q,
                    "event_dependency": "Single-Event",
                    "stationary": "YES",
                    "labels": build_final_labels(raw)
                })

        # ---- STATIONARY MULTI ----
        for q in qs_sm:
            raw = label_stationary(q)
            if raw:
                output.append({
                    "q_id": row["q_id"],
                    "question": q,
                    "event_dependency": "Multi-Event",
                    "stationary": "YES",
                    "labels": build_final_labels(raw)
                })

        # ---- NON-STATIONARY SINGLE ----
        for q in qs_ns:
            raw = label_nonstationary(q)
            if raw:
                output.append({
                    "q_id": row["q_id"],
                    "question": q,
                    "event_dependency": "Single-Event",
                    "stationary": "NO",
                    "labels": build_final_labels(raw)
                })

        # ---- NON-STATIONARY MULTI ----
        for q in qs_nm:
            raw = label_nonstationary(q)
            if raw:
                output.append({
                    "q_id": row["q_id"],
                    "question": q,
                    "event_dependency": "Multi-Event",
                    "stationary": "NO",
                    "labels": build_final_labels(raw)
                })

    # ---- write JSONL output ----
    with open(output_path, "w", encoding="utf-8") as f:
        for o in output:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print("Saved:", output_path)


########################################################
# 9. RUN
########################################################

if __name__ == "__main__":
    generate_recencyqa_4way("C:\\ws2025\\aktuelle\\RecencyQA\\NewDataset\\RecencyQA_dataset_small.json",
        "C:\\ws2025\\aktuelle\\RecencyQA\\generatedSet\\recencyqa_betterMulti.json",
        n_single_stationary=2,
        n_multi_stationary=2,
        n_single_nonstat=2,
        n_multi_nonstat=2,
    )
