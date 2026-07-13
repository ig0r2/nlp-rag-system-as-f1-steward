import json
from openai.types.chat import ChatCompletionUserMessageParam

from utils.logger import Logger
from utils.path import get_logs_path
from utils.rules import Rules
from utils.server import get_llm_client, COLLECTIONS


def predict_decision_single(query_metadata: dict, llm_client, llm_model, collection, rules: Rules, k: int) -> dict:
    query_text = query_metadata.get("description", "")
    filename = query_metadata.get("filename", "")

    # Top K+1 pa filtriramo self
    results = collection.query(
        query_texts=[query_text],
        n_results=k + 1,
        include=["metadatas", "distances"]
    )

    neighbors = [(m, d) for m, d in zip(results["metadatas"][0], results["distances"][0])
                 if m.get("filename") != filename][:k]

    context_cases = ""
    all_sections = []
    for i, (neighbor, distance) in enumerate(neighbors):
        similarity = 1 - distance
        # if similarity < 0.2:
        #     continue
        sections = neighbor.get("sections", [])
        context_cases += f"""
    Case {i + 1} (similarity: {similarity:.2f}):
    Session: {neighbor.get('session', 'N/A')} 
    Fact: {neighbor.get('fact', 'N/A')}
    Infringement sections: {', '.join(sections) if sections else 'N/A'}
    Decision: {neighbor.get('decision', 'N/A')}
    Reason: {neighbor.get('reason', 'N/A')}
    ---"""
        all_sections += sections

    all_sections_set = rules.get_related_rules(list(set(all_sections)), max_depth=2, max_total=15)

    context_rules = ""
    for section in all_sections_set:
        if rule_text := rules.get_rule(section):
            context_rules += rule_text + "\n"

    prompt = f"""You are an FIA steward assistant analyzing Formula 1 racing incidents.

    Below are {k} similar past cases with their decisions, including rules from rulebook related to those cases. 
    Based on these precedents and official rules, predict the decision for the new incident.

    SIMILAR PAST CASES:
    {context_cases}

    RELATED RULES:
    {context_rules}

    NEW INCIDENT:
    Session: {query_metadata.get('session', 'N/A')} 
    Description: {query_metadata.get('description', 'N/A')}

    Based on the similar cases and rules above, what decision would the stewards likely make?
    Consider the pattern of decisions in similar cases. 
    Consider the session it took place in. 
    Consider the rules.

    Respond in JSON with this exact format:
    {{
        "decision": "<predicted decision>",
        "reasoning": "<brief explanation based on the similar cases>"
        "rule_reference": "<cite the rules that were breached if they are available to you>"
        "rule_cite": "<cite the rules text of the exact rule part that was breached>"
    }}"""

    print("=" * 40, "Prompt", "=" * 40)
    print(prompt)

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    print("=" * 40, "Reasoning", "=" * 40)
    if "openai" in llm_model:
        print(response.choices[0].message.reasoning)
    else:
        print(response.choices[0].message.reasoning_content)
    print()

    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {"decision": raw, "reasoning": ""}

    print("=" * 40, "Raw Answer", "=" * 40)
    print(raw)

    return {
        "predicted_decision": parsed.get("decision", ""),
        "reasoning": parsed.get("reasoning", ""),
    }


def predict_single(meta, llm_model, k):
    filename = meta.get("filename", "")
    log_file = get_logs_path() / f"predicted_single/{filename}_K{k}.txt"

    llm_client = get_llm_client()
    collection = COLLECTIONS["default"].get_collection()
    rules = Rules()

    with Logger(log_file):
        result = predict_decision_single(meta, llm_client, llm_model, collection, rules, k)

        result["filename"] = meta.get("filename", "N/A")
        result["decision"] = meta.get("decision", "N/A")
        result["decision_n"] = meta.get("decision_n", "N/A")
        result["decision_category"] = meta.get("decision_category", "N/A")

        print("=" * 40, "Results", "=" * 40)
        print(f"Actual:    {meta.get("decision", "")} ({meta.get("decision_category", "")})")
        print(f"Predicted: {result['predicted_decision']}")
        print(f"Reasoning: {result['reasoning']}")
        print()


meta = {
    "filename": "2026 monaco rusel",
    "session": "Race",
    "description": "Car 63 did not serve 5 seconds penalty when doing a pitstop after getting the penalty. The pitstop was done under the safety car, the crew touched the car as soon as it entered the pitbox.",
}

# meta = {
#     "filename": "2026 monaco hajar",
#     "session": "Race",
#     "description": "During the suspension of the Race, mechanics of the Oracle Red Bull Racing Team were working on car number 6, performing operations not permitted by Article B5.14.4.a. at 16:55. When queried about their works, they stopped working and reverted the car to its previous state without replacing any part..",
# }

if __name__ == "__main__":
    K = 60

    # LLM_MODEL = "google/gemma-4-e2b"
    # LLM_MODEL = "google/gemma-4-e4b"
    # LLM_MODEL = "google/gemma-4-12b-qat"
    # LLM_MODEL = "openai/gpt-oss-20b"
    # LLM_MODEL = "qwen/qwen3.5-9b"
    LLM_MODEL = "qwen2.5-coder-7b-instruct"

    predict_single(meta, LLM_MODEL, K)
