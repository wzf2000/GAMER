import os
import json


def ensure_dir(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)


def load_json(file: str):
    with open(file, "r") as f:
        data = json.load(f)
    return data
