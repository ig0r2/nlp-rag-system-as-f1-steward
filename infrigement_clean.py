from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

import json

client = OpenAI(base_url="http://localhost:1234/v1", api_key="none")


def normalize_infringement(raw_text):
    response = client.chat.completions.create(
        model="qwen2.5-coder-7b-instruct",
        temperature=0,
        reasoning_effort="low",
        messages=[
            ChatCompletionUserMessageParam(
                role="user",
                content=f"""Extract all regulation references and normalize them to a strict format.

STRICT FORMAT RULES:
- Sporting Regulations articles: "SR Article 33.4"
- Technical Regulations articles: "TR Article 3.15"
- International Sporting Code: "ISC Article 12.2.1"
- Appendix L articles: "ISC Appendix L Chapter IV Article 2"  
- International Sporting Code Appendix H articles: "ISC Appendix H Article 12.2.1"

Ignore Race Director's Event Notes
Do not write letter subsections - .a | .b | a) | b) | c) and so on

EXAMPLES:
"Article 33.4 b) of the FIA Formula One Sporting Regulations" → "SR Article 33.4"
"Article 12.2.1 i) of the International Sporting Code" → "ISC Article 12.2.1"
"Appendix L, Chapter IV, Article 2 (b) of the International Sporting Code" → "ISC Appendix L Chapter IV Article 2"
"Appendix H, Article 2.10.15 b of the International Sporting Code" → "ISC Appendix H Article 2.10.15"
"Articles 28.2" → "SR Article 28.2"
"Article 3.15.17of the FIA Technical Regulations" → "TR Article 3.15.17"

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

    return json_text


def process_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for i, item in enumerate(dataset):
        raw = item.get("infringement", "")
        if not raw:
            print(f"[{i}] Preskočeno — nema 'infringement' polja")
            results.append({**item, "sections": []})
            continue

        refs = normalize_infringement(raw)
        print(f"[{i}]: {refs}")
        results.append({**item, "sections": refs})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Obrađeno {len(results)} stavki. Rezultati u {path}")


if __name__ == "__main__":
    process_dataset("data/json_raw/2022.json")
    process_dataset("data/json_raw/2023.json")
    # process_dataset("data/json/2024.json")
    # process_dataset("data/json/2025.json")
