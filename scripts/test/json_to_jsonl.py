import json

from utils.data import load_json_data
from utils.path import get_data_path

if __name__ == "__main__":
    paths = [
        get_data_path() / "predicted/2025/qwen3.5-9b_K10.json",
        get_data_path() / "predicted/2025/qwen3.5-9b_K20.json",
    ]

    for path in paths:
        data_input = load_json_data(path)
        with open(path.with_suffix(".jsonl"), "a") as out_f:
            for item in data_input:
                out_f.write(json.dumps(item) + "\n")
