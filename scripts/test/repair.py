from utils.data import load_json_data, write_json_data
from utils.path import get_data_path


def apply(data_in, data_out):
    count = 0
    for i in range(len(data_out)):
        if data_in[i]["filename"] == data_out[i]["filename"]:
            data_out[i]["decision_n"] = data_in[i]["decision_n"]
            data_out[i]["decision_category"] = data_in[i]["decision_category"]
            count += 1
        else:
            print(i, "Skipped")
    print("Done", count)


if __name__ == "__main__":
    paths = [
        # get_data_path() / "json/2022.json",
        # get_data_path() / "json/2023.json",
        # get_data_path() / "json/2024.json",
        get_data_path() / "json/2025.json"
    ]
    for path_input in paths:
        data_input = load_json_data(path_input)
        for path_output in (get_data_path() / "predicted/2025").glob("*.json"):
            data_output = load_json_data(path_output)
            apply(data_input, data_output)
            write_json_data(path_output, data_output)
