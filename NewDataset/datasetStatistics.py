import json
from collections import Counter
from statistics import mean

# Load dataset
with open("recencyqa_OUR_DATASET.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Helper to count words
def count_words(text):
    return len(text.split())

# Initialize counters
total_questions = len(data)
stationary_count = 0
non_stationary_count = 0
single_event_count = 0
two_event_count = 0
three_event_count = 0
multi_event_causal = 0
multi_event_temporal = 0
label_counts = []
question_lengths = []
context1_lengths = []
context2_lengths = []
unique_seed_questions = set()
all_labels = []
recency_label1_count = 0
recency_label2_count = 0

# Iterate through dataset
for item in data:
    unique_seed_questions.add(item.get("question", ""))
    
    # Stationary
    if item.get("stationary", "NO") == "YES":
        stationary_count += 1
    else:
        non_stationary_count += 1

    # Event counts
    num_events = item.get("num_events", 1)
    if num_events == 1:
        single_event_count += 1
    elif num_events == 2:
        two_event_count += 1
    elif num_events == 3:
        three_event_count += 1

    # Multi-event type
    if item.get("event_dependency") == "Multi-Event":
        if item.get("generation_type") == "causal":
            multi_event_causal += 1
        elif item.get("generation_type") == "temporal_only":
            multi_event_temporal += 1

    # Labels and context lengths
    labels = item.get("labels", [])
    label_counts.append(len(labels))
    for lbl in labels:
        if "recency_label1" in lbl:
            all_labels.append(lbl["recency_label1"])
            context1_lengths.append(count_words(lbl.get("context1", "")))
            recency_label1_count += 1
        if "recency_label2" in lbl:
            all_labels.append(lbl["recency_label2"])
            context2_lengths.append(count_words(lbl.get("context2", "")))
            recency_label2_count += 1

    # Question length
    question_lengths.append(count_words(item.get("question", "")))

# Calculate averages
avg_question_length = mean(question_lengths) if question_lengths else 0
avg_context1_length = mean(context1_lengths) if context1_lengths else 0
avg_context2_length = mean(context2_lengths) if context2_lengths else 0

# Most frequent labels (top 3)
top_labels = Counter(all_labels).most_common(3)

# Output
print(f"Total questions (all splits) & {total_questions} \\\\")
print(f"Unique seed questions & {len(unique_seed_questions)} \\\\")
print(f"Stationary questions & {stationary_count} \\\\")
print(f"Non-stationary questions & {non_stationary_count} \\\\")
print(f"Single-event questions & {single_event_count} \\\\")
print(f"Two-event questions & {two_event_count} \\\\")
print(f"Three-event questions & {three_event_count} \\\\")
print(f"Multi-event (causal) & {multi_event_causal} \\\\")
print(f"Multi-event (temporal-only) & {multi_event_temporal} \\\\")
print(f"1-label items & {label_counts.count(1)} \\\\")
print(f"2-label items & {label_counts.count(2)} \\\\")
print(f"Avg. question length (words) & {avg_question_length:.1f} \\\\")
print(f"Avg. context 1 length (words) & {avg_context1_length:.1f} \\\\")
print(f"Avg. context 2 length (words) & {avg_context2_length:.1f} \\\\")
print(f"recency_label1 count & {recency_label1_count} \\\\")
print(f"recency_label2 count & {recency_label2_count} \\\\")
print(f"Most frequent labels (top-3) & {top_labels} \\\\")