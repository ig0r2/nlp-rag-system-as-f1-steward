from utils.data import load_json_data
from utils.path import get_project_root


# Compare categorizing of original decision text vs categorizing of normalized decision text


def check(data1, data2):
    for i in range(len(data1)):
        if data1[i]["actual_category"] != data2[i]["actual_category"]:
            print(f"[{i}] {data1[i]["actual_decision"]} ({data1[i]["actual_category"]}) <-> "
                  f"{data2[i]["actual_decision_n"]} ({data2[i]["actual_category"]})")
        if data1[i]["predicted_category"] != data2[i]["predicted_category"]:
            print(f"[{i}] {data1[i]["predicted_decision"]} ({data1[i]["predicted_category"]}) <-> "
                  f"{data2[i]["predicted_decision_n"]} ({data2[i]["predicted_category"]})")


if __name__ == "__main__":
    data1 = load_json_data(get_project_root() / "logs/predict/nomic-v1.5_Q8_google_gemma-4-e4b_predicted_2.json")
    data2 = load_json_data(get_project_root() / "logs/predict/nomic-v1.5_Q8_google_gemma-4-e4b_n_2.json")

    check(data1, data2)
