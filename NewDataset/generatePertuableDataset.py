import json
from tqdm import tqdm
from together import Together

# ---- Together AI setup ----
client = Together()
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
print("Using Together AI model:", MODEL_NAME)

def llm(prompt, max_new_tokens=350):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
    )
    return response.choices[0].message.content

# ---- load/save JSONL ----
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # ignore empty lines
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ---- check perturbability + generate in one prompt ----
def check_and_generate_perturbation(question, labels, possible_labels, contexts):
    """
    Prüft, ob die Frage perturbierbar ist, und generiert direkt eine neue Frage, falls ja.
    Rückgabe: dict {"perturbable": True/False, "new_question": str oder None}
    """
    prompt = f"""
Question: "{question}"
Current labels: {labels}
Possible labels: {possible_labels}
Context: {contexts}

Task: 
1. Determine whether this question could have a different label if one or two words
   in the question or context were changed slightly.
2. If it is perturbable, generate 1 variation of the question by changing those words,
   keeping the overall meaning and grammar correct.

Return only JSON like:
{{
    "perturbable": true,
    "new_question": "<new question>"
}}
or
{{
    "perturbable": false,
    "new_question": null
}}
Do NOT include explanations or extra text.
"""
    text = llm(prompt)
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        js = json.loads(text[start:end])
        return js
    except:
        return {"perturbable": False, "new_question": None}

# ---- label the new question using your existing pipeline ----
def label_question(q, stationary=True):
    """
    Return a list of dicts like your build_final_labels output.
    """
    from generateDataset_multiEventLabel import label_stationary, label_nonstationary, build_final_labels
    if stationary:
        raw = label_stationary(q)
    else:
        raw = label_nonstationary(q)
    if raw:
        return build_final_labels(raw)
    return None

# ---- main augmentation ----
def augment_dataset(input_path, output_path):
    dataset = load_jsonl(input_path)
    augmented = []

    for item in tqdm(dataset):
        # Extract needed info
        question = item["question"]
        labels = [d.get("recency_label1") for d in item.get("labels", [])]
        possible_labels = [
            "An-Hour","A-Few-Hours","A-Day","A-Few-Days","A-Week","A-Few-Weeks",
            "A-Month","A-Few-Months","A-Year","A-Few-Years","Many-Years","Never"
        ]
        contexts = [d.get("context1") for d in item.get("labels", [])]

        # AI checks perturbability and generates new question if needed
        result = check_and_generate_perturbation(question, labels, possible_labels, contexts)

        # Add original item
        item_copy = item.copy()
        item_copy["perturbable"] = result["perturbable"]
        augmented.append(item_copy)

        # Generate new variant if perturbable
        if result["perturbable"] and result["new_question"]:
            stationary = item_copy.get("stationary", "YES") == "YES"
            new_labels = label_question(result["new_question"], stationary=stationary)
            if new_labels:
                new_item = item_copy.copy()
                new_item["question"] = result["new_question"]
                new_item["labels"] = new_labels
                new_item["perturbable"] = True
                augmented.append(new_item)

    save_jsonl(augmented, output_path)
    print(f"Augmented dataset saved to {output_path}")

# ---- run ----
if __name__ == "__main__":
    augment_dataset(
        input_path="C:\\ws2025\\aktuelle\\RecencyQA\\generatedSet\\recencyqa_betterMulti.json",
        output_path="C:\\ws2025\\aktuelle\\RecencyQA\\generatedSet\\recencyqa_augmented_ai.json"
    )
