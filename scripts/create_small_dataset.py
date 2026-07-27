"""
Extract a ~1/10 user subset from ShortVideoAD to create ShortVideoSmall.

Only user-keyed files are subsetted; item-level files (item.json, behavior_level.json)
are copied as-is since they cover the full item space.
"""

import json
import os
import pickle
import random
import shutil

SRC = "/home/zhouman/guoyunhe/workspace/full/GAMER/data/ShortVideoAD"
DST = "/home/zhouman/guoyunhe/workspace/full/GAMER/data/ShortVideoADSmall"
SRC_NAME = "ShortVideoAD"
DST_NAME = "ShortVideoADSmall"
RATIO = 0.01
SEED = 42

os.makedirs(DST, exist_ok=True)

# Load all user-keyed files
with open(f"{SRC}/{SRC_NAME}.SMB.inter.json") as f:
    inter = json.load(f)
with open(f"{SRC}/{SRC_NAME}.SMB.behavior.json") as f:
    behavior = json.load(f)
with open(f"{SRC}/{SRC_NAME}.SMB.session.json") as f:
    session = json.load(f)
with open(f"{SRC}/{SRC_NAME}.SMB.time.json") as f:
    time = json.load(f)
with open(f"{SRC}/{SRC_NAME}.index.json") as f:
    index = json.load(f)  # item index — keep full, not user-keyed
with open(f"{SRC}/{SRC_NAME}.SMB.data.pkl", "rb") as f:
    data = pickle.load(f)

all_uids = list(inter.keys())
n_select = max(1, round(len(all_uids) * RATIO))
rng = random.Random(SEED)
selected_uids = set(rng.sample(all_uids, n_select))
print(f"Total users: {len(all_uids)} → selected: {len(selected_uids)}")

def subset_dict(d):
    return {uid: d[uid] for uid in selected_uids if uid in d}

# Subset user-keyed JSON files
with open(f"{DST}/{DST_NAME}.SMB.inter.json", "w") as f:
    json.dump(subset_dict(inter), f)
with open(f"{DST}/{DST_NAME}.SMB.behavior.json", "w") as f:
    json.dump(subset_dict(behavior), f)
with open(f"{DST}/{DST_NAME}.SMB.session.json", "w") as f:
    json.dump(subset_dict(session), f)
with open(f"{DST}/{DST_NAME}.SMB.time.json", "w") as f:
    json.dump(subset_dict(time), f)

# index.json is item-keyed (item_id → token list), keep full like item.json
with open(f"{DST}/{DST_NAME}.index.json", "w") as f:
    json.dump(index, f)

# Subset data.pkl
new_data = {
    "session":    {uid: data["session"][uid]    for uid in selected_uids if uid in data["session"]},
    "train_pos":  {uid: data["train_pos"][uid]  for uid in selected_uids if uid in data["train_pos"]},
    "valid_pos":  {uid: data["valid_pos"][uid]  for uid in selected_uids if uid in data["valid_pos"]},
    "test_pos":   {uid: data["test_pos"][uid]   for uid in selected_uids if uid in data["test_pos"]},
    "time":       {uid: data["time"][uid]       for uid in selected_uids if uid in data["time"]},
}
with open(f"{DST}/{DST_NAME}.SMB.data.pkl", "wb") as f:
    pickle.dump(new_data, f)

# Copy item-level files unchanged
for fname in ["item.json", "behavior_level.json"]:  # index.json already written above
    src_file = f"{SRC}/{SRC_NAME}.{fname}"
    if os.path.exists(src_file):
        shutil.copy(src_file, f"{DST}/{DST_NAME}.{fname}")
        print(f"Copied {fname}")

print("Done.")
print(f"Output: {DST}/")
