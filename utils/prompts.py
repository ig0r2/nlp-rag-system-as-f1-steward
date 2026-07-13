import json

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from utils.rules import Rules


def normalize_decision(client, decision):
    system_prompt = f"""You are an expert in FIA Formula One regulations and stewards' decisions.

Your task is to normalize raw F1 stewards' incident decision text into a standardized, canonical form. The normalized output must:

1. Remove all references to specific people or entities:
   - No driver names (e.g. "Carlos Sainz", "Yuki Tsunoda")
   - No car numbers (e.g. "Car 14", "Car 20")
   - No team or competitor names (e.g. "Oracle Red Bull Racing", "Alfa Romeo F1 Team ORLEN", "McLaren")

2. Standardize phrasing and formatting:
   - Use consistent terminology across equivalent decisions
   - Normalize monetary amounts to a standard format: €[amount] (e.g. "€10,000")
   - Normalize penalty point references to: "[N] penalty point(s)." — omit any cumulative total
   - Remove redundant or parenthetical explanations that restate the obvious (e.g. "(10 seconds added to elapsed Race time)")
   - Normalize session references: use "Race", "Sprint", "Qualifying" (capitalised)

3. Resolve decision equivalences:
   - Treat "No Further Action", "No further action.", "No penalty applied", "No penalty is applied" as the same canonical decision
   - Treat "Warning", "Warning for the driver", "Formal warning to the driver and the team" as equivalent where appropriate
   - Treat "Reprimand (Driving)" and "A penalty of a Reprimand" as equivalent

4. Preserve all factual content:
   - Penalty durations, amounts, and conditions
   - Suspended fine conditions
   - Grid position conversions
   - Licence suspension details

5. Output only the normalized decision string. Do not include explanations, preamble, or metadata.

--- CANONICAL DECISION FORMATS ---
No penalty:
  No further action.

Time penalties:
  [N]-second time penalty. [X] penalty point(s).
  [N]-second stop-and-go penalty. [X] penalty point(s).
  Drive through penalty.

Grid penalties:
  Drop of [N] grid position for the next Race
  Required to start the [session] from the pit lane.
  Required to start the [session] from the back of the grid.
  [N]-second time penalty, converted to [N] grid position drop for next Race.

Disqualification:
  Disqualified from the [session] classification.

Fines:
  Fine of €[amount].
  Fine of €[amount] - €[suspended amount] suspended.

Warnings / Reprimands:
  Warning.
  Formal warning to the driver.
  Formal warning to the driver. Competitor fined €[amount].
  Team reprimand.
  Reprimand (Driving).

Licence:
  Super Licence suspended for next Competition. [N] penalty points removed.

Other:
  Obligation to perform public interest work.
---"""

    user_prompt = f"""
Normalize the following F1 stewards' decision:

{decision}

Return only the normalized decision string."""

    # qwen2.5-coder-7b-instruct
    # google/gemma-4-e4b
    response = client.chat.completions.create(
        model="qwen2.5-coder-7b-instruct",
        temperature=0,
        messages=[ChatCompletionSystemMessageParam(role="system", content=system_prompt),
                  ChatCompletionUserMessageParam(role="user", content=user_prompt)]
    )

    return response.choices[0].message.content


def get_decision_category(client, decision):
    prompt = f"""You are categorizing F1 steward decision.

    Decision: {decision}

    Categories:
    no_action = no further action (standalone, no penalty at all) - examples: "No further investigaion."
    warning = warning, ignore penalty points - examples: "Warning.", "Driver Warning.", "Warning. 1 penalty point."
    reprimand = reprimand, ignore penalty points - examples: "Reprimand.", "Reprimand (Driving).", "Reprimand. 1 penalty point."
    fine = financial fine
    penalty_points = only if penalty points only
    time_penalty = time penalty (5s, 10s, stop-go, drive-through), ignore penalty points 
    grid_penalty = grid penalty (back of grid, pit lane start) - examples: "Drop of X grid positions for the next Race", "Required to start the Race from the pit lane."
    dsq = disqualification
    race_suspension = - examples: "Suspended from the next race", "Super Licence suspended for next Competition."

    Return only category name"""

    # qwen2.5-coder-7b-instruct
    # google/gemma-4-e4b
    response = client.chat.completions.create(
        model="qwen2.5-coder-7b-instruct",
        messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def check_decision_equality(client, actual, predicted):
    prompt = f"""You are evaluating F1 steward decisions. Check if these 2 decision are the same penalty or not.

        Decision 1: {actual}
        Decision 2: {predicted}

        Return true/false only"""

    # qwen2.5-coder-7b-instruct
    # google/gemma-4-e4b
    response = client.chat.completions.create(
        model="qwen2.5-coder-7b-instruct",
        messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    return True if raw.lower() == "true" else False


def predict_decision(query_metadata: dict, llm_client, llm_model, collection, rules: Rules, k: int) -> dict:
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

    Respond with valid JSON only. Escape any double quotes inside string values with a backslash: \\"
    Respond with this exact format:
    {{
        "decision": "<predicted decision>",
        "reasoning": "<brief explanation based on the similar cases>"
    }}"""

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
        temperature=0,
    )

    # --- Extract and Print Token Usage ---
    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        print("\n=== Token Usage ===")
        print(f"Prompt Tokens:     {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        print(f"Total Tokens:      {total_tokens}")
        print("===================\n")
    else:
        print("\n[Warning] Usage data not found in the response.\n")

    raw = response.choices[0].message.content.strip()

    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {"decision": raw, "reasoning": ""}

    return {
        "predicted_decision": parsed.get("decision", ""),
        "reasoning": parsed.get("reasoning", ""),
    }


def predict_decision_no_rules(query_metadata: dict, llm_client, llm_model, collection, rules: Rules, k: int) -> dict:
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

    prompt = f"""You are an FIA steward assistant analyzing Formula 1 racing incidents.

    Below are {k} similar past cases with their decisions. 
    Based on these precedents, predict the decision for the new incident.

    SIMILAR PAST CASES:
    {context_cases}

    NEW INCIDENT:
    Session: {query_metadata.get('session', 'N/A')} 
    Description: {query_metadata.get('description', 'N/A')}

    Based on the similar cases, what decision would the stewards likely make?
    Consider the pattern of decisions in similar cases. 
    Consider the session it took place in. 

    Respond with valid JSON only. Escape any double quotes inside string values with a backslash: \\"
    Respond with this exact format:
    {{
        "decision": "<predicted decision>",
        "reasoning": "<brief explanation based on the similar cases>"
    }}"""

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
        temperature=0,
    )

    # --- Extract and Print Token Usage ---
    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        print("\n=== Token Usage ===")
        print(f"Prompt Tokens:     {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        print(f"Total Tokens:      {total_tokens}")
        print("===================\n")
    else:
        print("\n[Warning] Usage data not found in the response.\n")

    raw = response.choices[0].message.content.strip()

    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {"decision": raw, "reasoning": ""}

    return {
        "predicted_decision": parsed.get("decision", ""),
        "reasoning": parsed.get("reasoning", ""),
    }




def predict_decision_no_cases(query_metadata: dict, llm_client, llm_model, collection, rules: Rules, k: int) -> dict:
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

    all_sections = []
    for i, (neighbor, distance) in enumerate(neighbors):
        sections = neighbor.get("sections", [])
        all_sections += sections

    all_sections_set = rules.get_related_rules(list(set(all_sections)), max_depth=2, max_total=15)

    context_rules = ""
    for section in all_sections_set:
        if rule_text := rules.get_rule(section):
            context_rules += rule_text + "\n"

    prompt = f"""You are an FIA steward assistant analyzing Formula 1 racing incidents.

    Below are {k} similar past cases with their decisions, including rules from rulebook related to those cases. 
    Based on these precedents and official rules, predict the decision for the new incident.

    RELATED RULES:
    {context_rules}

    NEW INCIDENT:
    Session: {query_metadata.get('session', 'N/A')} 
    Description: {query_metadata.get('description', 'N/A')}

    Based on the similar cases and rules above, what decision would the stewards likely make?
    Consider the pattern of decisions in similar cases. 
    Consider the session it took place in. 
    Consider the rules.

    Respond with valid JSON only. Escape any double quotes inside string values with a backslash: \\"
    Respond with this exact format:
    {{
        "decision": "<predicted decision>",
        "reasoning": "<brief explanation based on the similar cases>"
    }}"""

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
        temperature=0,
    )

    # --- Extract and Print Token Usage ---
    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        print("\n=== Token Usage ===")
        print(f"Prompt Tokens:     {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        print(f"Total Tokens:      {total_tokens}")
        print("===================\n")
    else:
        print("\n[Warning] Usage data not found in the response.\n")

    raw = response.choices[0].message.content.strip()

    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {"decision": raw, "reasoning": ""}

    return {
        "predicted_decision": parsed.get("decision", ""),
        "reasoning": parsed.get("reasoning", ""),
    }



def predict_decision_nothing(query_metadata: dict, llm_client, llm_model, collection, rules: Rules, k: int) -> dict:

    prompt = f"""You are an FIA steward assistant analyzing Formula 1 racing incidents.

    Below are {k} similar past cases with their decisions. 
    Based on these precedents, predict the decision for the new incident.

    NEW INCIDENT:
    Session: {query_metadata.get('session', 'N/A')} 
    Description: {query_metadata.get('description', 'N/A')}

    Based on the similar cases, what decision would the stewards likely make?
    Consider the pattern of decisions in similar cases. 
    Consider the session it took place in. 

    Respond with valid JSON only. Escape any double quotes inside string values with a backslash: \\"
    Respond with this exact format:
    {{
        "decision": "<predicted decision>",
        "reasoning": "<brief explanation based on the similar cases>"
    }}"""

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
        temperature=0,
    )

    # --- Extract and Print Token Usage ---
    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        print("\n=== Token Usage ===")
        print(f"Prompt Tokens:     {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        print(f"Total Tokens:      {total_tokens}")
        print("===================\n")
    else:
        print("\n[Warning] Usage data not found in the response.\n")

    raw = response.choices[0].message.content.strip()

    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {"decision": raw, "reasoning": ""}

    return {
        "predicted_decision": parsed.get("decision", ""),
        "reasoning": parsed.get("reasoning", ""),
    }
