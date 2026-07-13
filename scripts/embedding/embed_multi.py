from utils.data import load_json_data
from utils.path import get_data_path
from utils.server import DB, COLLECTIONS

ITEMS: list[DB] = [
    COLLECTIONS["minilm"],
    COLLECTIONS["gemma"],
    COLLECTIONS["qwen3-4B"],
    COLLECTIONS["bge-large"],
    COLLECTIONS["mxbai"],
    COLLECTIONS["nomic-v1.5"],
    COLLECTIONS["nomic-v1.5_Q8"],
    COLLECTIONS["nomic-v2"],
]

##############################################


incidents = []
for path in (get_data_path() / "json").glob("*.json"):
    for el in load_json_data(path):
        incidents.append({
            "document": el["fact"] + " " + el["reason"],
            "metadata": {**el}
        })

for item in ITEMS:
    collection = item.get_or_create_collection()

    # Clear
    if all_ids := collection.get()["ids"]:
        collection.delete(ids=all_ids)

    # Add
    collection.add(
        documents=[i["document"] for i in incidents],
        metadatas=[i["metadata"] for i in incidents],
        ids=[str(idx) for idx in range(len(incidents))]
    )

    print(collection.count())
