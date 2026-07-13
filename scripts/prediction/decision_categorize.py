from utils.data import load_json_data, write_json_data
from utils.path import get_data_path
from utils.prompts import get_decision_category, check_decision_equality
from utils.server import get_llm_client


def dataset_decision_categorize(path):
    # load json
    data = load_json_data(path)

    # process via llm
    llm_client = get_llm_client()
    for i, item in enumerate(data):
        actual = item["decision_n"]
        actual_category = item["decision_category"]
        predicted = item["predicted_decision_n"]

        predicted_category = get_decision_category(llm_client, predicted)

        equal = check_decision_equality(llm_client, actual,
                                        predicted) if actual_category == predicted_category else False

        item["predicted_decision_category"] = predicted_category
        item["equal"] = equal

        print(f"[{i + 1}] "
              f"actual: {actual[:40]} ({actual_category}) | "
              f"predicted: {predicted[:40]} ({predicted_category}) | "
              f"equal: {equal}")

    # save to same json file
    write_json_data(path, data)
    return path


if __name__ == "__main__":
    path = get_data_path() / "predicted/2025/gemma4-e4b_K10.json"

    for path in (get_data_path() / "predicted/2025/").glob("*.json"):
        dataset_decision_categorize(path)
