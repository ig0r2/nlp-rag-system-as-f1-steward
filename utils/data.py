import json
from pathlib import Path


def load_json_data(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_data(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        print("Saved to", path)
