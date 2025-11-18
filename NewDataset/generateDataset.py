import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

###############################################
# SETUP: Lokales Modell laden
###############################################

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"   # funktioniert in 8GB (4-bit)
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded:", MODEL_NAME)

###############################################
# Utility: LLM Chat Wrapper
###############################################

def llm(prompt, max_new_tokens=300, temp=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

###############################################
# Step 1 — Events extrahieren aus context1 / context2
###############################################

EVENT_EXTRACTION_PROMPT = """
Convert the following context description into explicit event statements.
Return 1–3 clear, factual, standalone events. Avoid narrative style.

Context:
{context}

Return JSON:
{{
 "events": ["event1", "event2", "event3"]
}}
"""

def extract_events(context):
    if context is None or context.strip() == "":
        return []
    prompt = EVENT_EXTRACTION_PROMPT.format(context=context)
    out = llm(prompt)
    try:
        j = json.loads(out[out.index("{"):out.rindex("}")+1])
        return j["events"]
    except:
        return [context]


###############################################
# Step 2 — Neue Fragen aus Events generieren
###############################################

QUESTION_GEN_PROMPT = """
Generate 5 temporal questions whose answers depend on the given events.
The questions must require time-based change or evolving conditions.

Events:
{events}

Return JSON:
{{
 "questions": ["q1", "q2", "q3", "q4", "q5"]
}}
"""

def generate_single_event_questions(events):
    prompt = QUESTION_GEN_PROMPT.format(events=events)
    out = llm(prompt)
    try:
        j = json.loads(out[out.index("{"):out.rindex("}")+1])
        return j["questions"]
    except:
        return []


###############################################
# Step 3 — Multi-Event temporale Fragen generieren
###############################################

MULTI_EVENT_PROMPT = """
Generate 5 multi-event temporal reasoning questions.
Each question must depend on at least two events and require ordering, causal,
or evolving relationships between events.

Events:
{events}

Return JSON:
{{
 "questions": ["q1", "q2", "q3", "q4", "q5"]
}}
"""

def generate_multi_event_questions(events):
    prompt = MULTI_EVENT_PROMPT.format(events=events)
    out = llm(prompt)
    try:
        j = json.loads(out[out.index("{"):out.rindex("}")+1])
        return j["questions"]
    except:
        return []


###############################################
# Step 4 — Labeling (Recency, Stationarity, Reasoning Type)
###############################################

LABEL_PROMPT = """
For the following question:

"{question}"

Assign:
1. recency_label
2. stationarity: Stationary / Non-Stationary
3. event_dependency: Single-Event or Multi-Event


Return JSON:
{{
 "recency": "...",
 "stationarity": "...",
 "event_dependency": "...",
 "explanation": "..."
}}
"""

def label_question(q, is_multi):
    prompt = LABEL_PROMPT.format(question=q)
    if is_multi:
        prompt += "\n(event_dependency should be Multi-Event)\n"
    out = llm(prompt)
    try:
        j = json.loads(out[out.index("{"):out.rindex("}")+1])
        return j
    except:
        return None


###############################################
# Step 5 — OPTIONAL: Gemini Validation
###############################################

USE_GEMINI = False

def validate_with_gemini(sample):
    # You can fill this with your Gemini Free Tier API later
    return {"valid": True}


###############################################
# MAIN — RecencyQA++ Generator
###############################################

def generate_recencyqa_plus(input_path, output_path):

    df = pd.read_json(input_path, lines=True)
    output = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        
        # 1. Events extrahieren
        events1 = extract_events(row["labels"][0]["context1"])
        events2 = []
        if row.get("stationary") == "NO":
            events2 = extract_events(row["labels"][1]["context2"])

        all_events = list(set(events1 + events2))

        # 2. Single-event Fragen generieren
        qs_single = generate_single_event_questions(all_events)

        # 3. Multi-event Fragen generieren
        qs_multi = []
        if len(all_events) >= 2:
            qs_multi = generate_multi_event_questions(all_events)

        # 4. Labeln
        for q in qs_single:
            labels = label_question(q, is_multi=False)
            if labels:
                sample = {
                    "source_qid": row["q_id"],
                    "question": q,
                    "events": all_events,
                    "labels": labels
                }
                output.append(sample)

        for q in qs_multi:
            labels = label_question(q, is_multi=True)
            if labels:
                sample = {
                    "source_qid": row["q_id"],
                    "question": q,
                    "events": all_events,
                    "labels": labels
                }
                output.append(sample)

    # Speichern
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done! Saved to", output_path)


###############################################
# RUN
###############################################

if __name__ == "__main__":
    generate_recencyqa_plus("RecencyQA_dataset_small.json", "recencyqa_plus.json")
