from scripts.prediction.decision_categorize import dataset_decision_categorize
from scripts.prediction.decision_normalize import dataset_decision_normalize
from scripts.prediction.evaluate import evaluate
from scripts.prediction.predict import predict_dataset

if __name__ == "__main__":
    K = 60

    # LLM_MODEL = "google/gemma-4-e2b"
    # OUT_NAME = "gemma4-e2b"

    LLM_MODEL = "gemma-4-e2b-it-qat"
    OUT_NAME = "gemma4-e2b-no-reasoning"

    # LLM_MODEL = "google/gemma-4-e4b"
    # OUT_NAME = "gemma4-e4b"

    # LLM_MODEL = "google/gemma-4-e4b-it-qat"
    # OUT_NAME = "gemma4-e4b-no-reasoning"

    # LLM_MODEL = "google/gemma-4-12b-qat"
    # OUT_NAME = "gemma4-12b"

    # LLM_MODEL = "openai/gpt-oss-20b"
    # OUT_NAME = "gpt-oss-20b_low"

    # LLM_MODEL = "qwen/qwen3.5-9b"
    # OUT_NAME = "qwen3.5-9b"

    # LLM_MODEL = "qwen2.5-coder-7b-instruct"
    # OUT_NAME = "qwen2.5-coder-7b"

    # OUT_NAME += "-no-cases-no-rules"
    # OUT_NAME += "-no-rules"
    # OUT_NAME += "-no-cases"

    evaluate(dataset_decision_categorize(dataset_decision_normalize(predict_dataset(LLM_MODEL, OUT_NAME, K))))
