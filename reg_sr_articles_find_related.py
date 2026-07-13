import re

from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

import json

client = OpenAI(base_url="http://localhost:1234/v1", api_key="none")


def normalize_infringement(raw_text):
    response = client.chat.completions.create(
        model="qwen2.5-coder-7b-instruct",
        temperature=0,
        max_tokens=2000,
        messages=[
            ChatCompletionUserMessageParam(
                role="user",
                content=f"""This is a text of regulation for a given Article. Extract all other regulation (Article) references and normalize them to a strict format.

EXAMPLES:
"... scrutineering in accordance with Article 31.1 will be deemed ..." → "Article 31.1"
"... from the start signal referred to in Article 44.10 to the end-of-session signal referred to in Article 59.1, shall be equal ..." → "["Article 44.10", "Article 59.1"]"
"... be suspended (see Article 57) the length ... " → "Article 57"
"... For testing carried out under Articles 10.8d) and 10.8e) ... " → "Article 10.8"
"... Any penalty imposed under Articles 43.5 (a), 44.4 or 58.3 ii) ... "  → "["Article 43.5", "Article 44.4", "Article 58.3"]"

Input: {raw_text}

Return ONLY a JSON list of normalized references. No other text."""
            )]
    )
    output = response.choices[0].message.content
    json_text = []
    try:
        cleaned = output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        json_text = json.loads(cleaned)
    except Exception as e:
        print(f"Greška: {e}\n{output}")

    # extract only numbers from output
    res = []
    for item in set(json_text):
        if "Article" not in item:
            continue
        matches = re.findall(r'\d+\.?\d*$', item)
        res += [f"SR Article {m}" for m in matches]

    return res


def process_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        sections = json.load(f)

    results = {}
    # Use .items() to iterate through the ID and the section data
    for section_id, section in sections.items():
        if "text" not in section:  # Simpler check than .keys()
            results[section_id] = {**section, "related": []}
            continue
        text = section["text"]

        refs = normalize_infringement(text)
        # Using section_id in the print statement instead of index 'i'
        print(f"[{section_id}]: {refs}")
        results[section_id] = {**section, "related": refs}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Obrađeno {len(results)} stavki. Rezultati u {path}")


if __name__ == "__main__":
    process_dataset("data/regulations/json/combined.json")
