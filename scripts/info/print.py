from utils.data import load_json_data, write_json_data
from utils.path import get_data_path

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def color_bool(value):
    color = GREEN if value else RED
    return f"{color}{value}{RESET}"

def apply(data):
    for i, item in enumerate(data):
        if i + 1 < 0: continue
        print(f"[{i + 1}] Decision: {item["predicted_decision"]}")
        print(f"      Normaliz: {item["predicted_decision_n"]}")
        print(f"      Category: {item["predicted_decision_category"]}")
        print(f"      Equal   : {color_bool(item['equal'])}")
        print(f"      Decision: {item["decision"]}")
        print(f"      Normaliz: {item["decision_n"]}")
        print(f"      Category: {item["decision_category"]}")
        print()


if __name__ == "__main__":
    paths = [
        # get_data_path() / "json/2022.json",
        # get_data_path() / "json/2023.json",
        # get_data_path() / "json/2024.json",
        # get_data_path() / "json/2025.json".
        get_data_path() / "predicted/2025/gemma4-e4b_K10.json"
    ]
    for path in paths:
        data = load_json_data(path)
        apply(data)
        write_json_data(path, data)
