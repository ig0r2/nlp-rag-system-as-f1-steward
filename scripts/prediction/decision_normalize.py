from pathlib import Path

import json

from utils.data import write_json_data, load_json_data
from utils.path import get_data_path
from utils.prompts import normalize_decision
from utils.server import get_llm_client

client = get_llm_client()


# for pipeline (predit returns jsonl)
def dataset_decision_normalize(path: Path):
    # load from jsonl
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError:
                pass

    # process via llm
    for i, item in enumerate(data):
        decision = item.get("predicted_decision", "")
        decision_n = normalize_decision(client, decision)
        item["predicted_decision_n"] = decision_n
        print("-" * 30)
        print(f"[{i + 1}] {decision}")
        print(f"      {decision_n}")

    # save to json
    output_path = path.with_suffix(".json")
    write_json_data(output_path, data)
    return output_path


# for manual json normalization
def dataset_decision_normalize_json(path: Path):
    # load from json
    data = load_json_data(path)

    # process via llm
    for i, item in enumerate(data):
        decision = item.get("predicted_decision", "")
        decision_n = normalize_decision(client, decision)
        item["predicted_decision_n"] = decision_n
        print("-" * 30)
        print(f"[{i + 1}] {decision}")
        print(f"      {decision_n}")

    # save to json
    write_json_data(path, data)
    return path


if __name__ == "__main__":
    # path = get_data_path() / "predicted/2025/qwen2.5-coder-7b_K40.jsonl"
    # dataset_decision_normalize(path)

    path = get_data_path() / "predicted/2025/qwen2.5-coder-7b_K5.json"
    paths = [
        get_data_path() / "predicted/2025/qwen2.5-coder-7b_K10.json",
        get_data_path() / "predicted/2025/qwen2.5-coder-7b_K20.json",
        get_data_path() / "predicted/2025/qwen2.5-coder-7b_K30.json",
        get_data_path() / "predicted/2025/qwen2.5-coder-7b_K40.json",
        get_data_path() / "predicted/2025/qwen2.5-coder-7b_K60.json",
        get_data_path() / "predicted/2025/qwen3.5-9b_K10.json",
    ]
    for path in paths:
        dataset_decision_normalize_json(path)
