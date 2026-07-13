from openai.types.chat import ChatCompletionUserMessageParam

from utils.data import load_json_data, write_json_data
from utils.path import get_data_path
from utils.server import get_llm_client


def clean(llm_client, text):
    prompt = f"""You are cleaning F1 steward incident document. 
    Remove any references of incident decisions/penalties. 
    Remove any references of regulation articles or sections. 
    The output should be only the incident description of factual events of that incident and should not contain steward decisions.

    Description: {text}

    Return only description text"""

    # qwen2.5-coder-7b-instruct
    # google/gemma-4-e4b
    response = llm_client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def apply(data):
    llm_client = get_llm_client()

    for i, item in enumerate(data):
        text = item["fact"] + " " + item["reason"]
        item["description"] = clean(llm_client, text)
        print(f"[{i + 1}]")


if __name__ == "__main__":
    paths = [
        get_data_path() / "json/2022.json",
        get_data_path() / "json/2023.json",
        get_data_path() / "json/2024.json",
        get_data_path() / "json/2025.json"
    ]
    for path in paths:
        data = load_json_data(path)
        apply(data)
        write_json_data(path, data)
