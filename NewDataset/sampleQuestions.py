import pandas as pd
import numpy as np

# Load full dataset
df = pd.read_json("RececnyQA_dataset.json")

# Extract primary recency label from labels[0]
def get_primary_recency(row):
    try:
        return row["labels"][0].get("recency_label1")
    except:
        return None

df["primary_recency"] = df.apply(get_primary_recency, axis=1)

# Split into stationary YES / NO
df_yes = df[df["stationary"] == "YES"]
df_no  = df[df["stationary"] == "NO"]

TARGET_TOTAL = 75
TARGET_YES = TARGET_TOTAL // 2        # 37
TARGET_NO  = TARGET_TOTAL - TARGET_YES  # 38

def stratified_sample(group_df, target_n):
    groups = group_df.groupby("primary_recency")
    samples = []

    total = len(group_df)

    # allocate proportional counts
    for name, grp in groups:
        frac = len(grp) / total
        n = max(1, int(frac * target_n))
        samples.append(grp.sample(n=n, random_state=42))

    out = pd.concat(samples)

    # adjust to exact count
    if len(out) > target_n:
        out = out.sample(target_n, random_state=42)
    elif len(out) < target_n:
        remainder = target_n - len(out)
        out = pd.concat([
            out,
            group_df.drop(out.index).sample(remainder, random_state=42)
        ])

    return out


# Sample separately
sample_yes = stratified_sample(df_yes, TARGET_YES)
sample_no  = stratified_sample(df_no, TARGET_NO)

# Combine final sample
df_selected = pd.concat([sample_yes, sample_no])
df_selected = df_selected.sample(frac=1, random_state=42)  # shuffle

df_selected.to_json("selected_75.json", orient="records", indent=2)

print("Final sample size:", len(df_selected))
