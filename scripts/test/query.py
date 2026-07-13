from utils.server import get_collection

# MODEL = "text-embedding-all-minilm-l6-v2-embedding"
# COLLECTION = "minilm"
# MODEL = "text-embedding-embeddinggemma-300m-qat"
# COLLECTION = "gemma"
# MODEL = "text-embedding-qwen3-embedding-4b"
# COLLECTION = "qwen3-4B"
MODEL = "text-embedding-nomic-embed-text-v1.5@q8_0"
COLLECTION = "nomic-v1.5_Q8"

##############################################

collection = get_collection(COLLECTION, MODEL)

results = collection.query(
    query_texts=["""Jewelery"""],
    n_results=5
)

for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- Result {i + 1} ---")
    print(results["metadatas"][0][i]["filename"])
    print("Fact:", results["metadatas"][0][i]["fact"])
    print("Infringement:", results["metadatas"][0][i]["infringement"])
    print("Decision:", results["metadatas"][0][i]["decision"])
