import json

import os

from utils.logger import Logger
from utils.path import get_data_path, get_logs_path
from utils.prompts import predict_decision, predict_decision_nothing, predict_decision_no_rules, \
    predict_decision_no_cases
from utils.rules import Rules
from utils.server import get_llm_client, COLLECTIONS


def check_completed(file):
    # Load already-processed filenames
    seen = set()
    if os.path.exists(file):
        with open(file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    seen.add(entry["filename"])
                except json.JSONDecodeError:
                    pass
        print(f"Resuming — {len(seen)} already done, skipping those.")
    return seen


def predict_dataset(llm_model, out_name, k):
    output_file = get_data_path() / f"predicted/2025/{out_name}_K{k}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file = get_logs_path() / f"predicted/2025/{out_name}_K{k}.txt"

    llm_client = get_llm_client()
    collection = COLLECTIONS["default"].get_collection()
    all_docs = collection.get(include=["metadatas", "documents"], where={"year": "2025"})
    metadatas = all_docs["metadatas"]
    total = len(metadatas)
    print(total)
    rules = Rules()

    seen = check_completed(output_file)
    with open(output_file, "a") as out_f:
        with Logger(log_file):
            for i, meta in enumerate(metadatas):
                filename = meta.get("filename", "")
                if filename in seen:
                    print(f"[{i + 1}/{total}] Skipping {filename}")
                    continue

                print(f"[{i + 1}/{total}] {filename}")
                result = predict_decision(meta, llm_client, llm_model, collection, rules, k)

                result["filename"] = meta.get("filename", "N/A")
                result["decision"] = meta.get("decision", "N/A")
                result["decision_n"] = meta.get("decision_n", "N/A")
                result["decision_category"] = meta.get("decision_category", "N/A")

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                print(f"Actual:    {result['decision']}")
                print(f"Predicted: {result['predicted_decision']}")
                print(f"Reasoning: {result['reasoning']}")
                print()

    print(f"Results saved to {output_file}")
    return output_file


if __name__ == "__main__":
    K = 40

    # LLM_MODEL = "google/gemma-4-e2b"
    # OUT_NAME = "gemma4-e2b"
    # OUT_NAME = "gemma4-e2b-2025"

    # LLM_MODEL = "google/gemma-4-e4b"
    # OUT_NAME = "gemma4-e4b"

    # LLM_MODEL = "openai/gpt-oss-20b"
    # OUT_NAME = "gpt-oss-20b_low"

    # LLM_MODEL = "qwen/qwen3.5-9b"

    LLM_MODEL = "qwen2.5-coder-7b-instruct"
    OUT_NAME = "qwen2.5-coder-7b"

    predict_dataset(LLM_MODEL, OUT_NAME, K)
