import chromadb
from openai import OpenAI
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.api import ClientAPI, Collection

from utils.path import get_db_path


def get_llm_client():
    return OpenAI(base_url="http://localhost:1234/v1", api_key="none")


class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = get_llm_client()

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.embeddings.create(model=self.model_name, input=input)
        return [item.embedding for item in response.data]


def _getdb__client() -> ClientAPI:
    return chromadb.PersistentClient(path=get_db_path())


# def get_collection(collection_name, model_name=None) -> Collection:
#     model_func = LocalEmbeddingFunction(model_name) if model_name is not None else None
#     return _getdb__client().get_collection(name=collection_name, embedding_function=model_func)


class DB:
    def __init__(self, id, name, model):
        self.id = id
        self.name = name
        self.model = model

    def get_or_create_collection(self) -> Collection:
        return _getdb__client().get_or_create_collection(name=self.id,
                                                         embedding_function=LocalEmbeddingFunction(self.model))

    def get_collection(self) -> Collection:
        return _getdb__client().get_collection(name=self.id, embedding_function=LocalEmbeddingFunction(self.model))


COLLECTIONS = {
    "default": DB("nomic-v1.5_Q8", "nomic-embed-text-v1.5_Q8", "text-embedding-nomic-embed-text-v1.5@q8_0"),
    "minilm": DB("minilm", "all-minilm-l6-v2-embedding_Q8", "text-embedding-all-minilm-l6-v2-embedding"),
    "gemma": DB("gemma", "embeddinggemma-300m-qat_Q4", "text-embedding-embeddinggemma-300m-qat"),
    "qwen3-4B": DB("qwen3-4B", "qwen3-embedding-4b_Q4_K_M", "text-embedding-qwen3-embedding-4b"),
    "bge-large": DB("bge-large", "bge-large-en-v1.5_Q8", "text-embedding-bge-large-en-v1.5"),
    "mxbai": DB("mxbai", "mxbai-embed-large-v1_F16", "text-embedding-bge-large-en-v1.5"),
    "nomic-v1.5": DB("nomic-v1.5", "nomic-embed-text-v1.5_Q4_K_M", "text-embedding-nomic-embed-text-v1.5@q4_k_m"),
    "nomic-v1.5_Q8": DB("nomic-v1.5_Q8", "nomic-embed-text-v1.5_Q8", "text-embedding-nomic-embed-text-v1.5@q8_0"),
    "nomic-v2": DB("nomic-v2", "nomic-embed-text-v2_Q8", "text-embedding-nomic-embed-text-v2-moe"),
}
