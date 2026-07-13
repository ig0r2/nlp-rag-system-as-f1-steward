import json

from utils.path import get_data_path

dataset = []
for path in (get_data_path() / "json").glob("*.json"):
    with open(path, "r", encoding="utf-8") as f:
        dataset += json.load(f)

unique_sections = sorted(set(
    item
    for m in dataset if "sections" in m
    for item in m["sections"]
))

print("\n", "=" * 50, "sections", "=" * 50, "\n")
for inf in unique_sections:
    print(inf)

unique_infringements = sorted(set(
    m["infringement"] for m in dataset if "infringement" in m
))

print("\n", "=" * 50, "infringement", "=" * 50, "\n")
for inf in unique_infringements:
    print(inf)

unique_decisions = sorted(set(
    m["decision"] for m in dataset if "decision" in m
))

print("\n", "=" * 50, "decision", "=" * 50, "\n")
for dec in unique_decisions:
    print(dec)
