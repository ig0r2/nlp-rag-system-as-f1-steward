from utils.data import load_json_data, write_json_data
from utils.path import get_data_path
from utils.prompts import get_decision_category
from utils.server import get_llm_client


def apply(data):
    llm_client = get_llm_client()

    for i, item in enumerate(data):
        decision = item["decision_n"]
        decision_category = get_decision_category(llm_client, decision)
        item["decision_category"] = decision_category
        print(f"[{i + 1}] "
              f"Decision: {decision[:40]} | Category: {decision_category}")


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
