from utils.data import write_json_data, load_json_data
from utils.path import get_data_path
from utils.prompts import normalize_decision
from utils.server import get_llm_client

client = get_llm_client()


def apply(data):
    for i, item in enumerate(data):
        decision = item.get("decision", "")
        decision_n = normalize_decision(client, decision)
        item["decision_n"] = decision_n
        print("-" * 30)
        print(f"[{i + 1}] {decision}")
        print(f"      {decision_n}")


if __name__ == "__main__":
    paths = [
        get_data_path() / "json/2022.json",
        get_data_path() / "json/2023.json",
        get_data_path() / "json/2024.json",
        get_data_path() / "json/2025.json"
    ]
    for path in paths:
        data = load_json_data(path)
        apply(data)
        write_json_data(path, data)
