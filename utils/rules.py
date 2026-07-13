import json

from utils.path import get_project_root


class Rules:
    rules: dict

    def __init__(self):
        self.rules = {}
        with open(get_project_root() / "data/regulations/json/combined.json", "r", encoding="utf-8") as f:
            self.rules = json.load(f)

    def get_rule(self, id):
        if (rule := self.rules.get(id)) is not None:
            return id + " " + rule['title'] + "\n" + (rule['text'] if 'text' in rule else "")
        return ""

    def get_related_rules(self, sections: list[str], max_depth: int = 1, max_total: int = 10) -> set[str]:
        """BFS expansion of related rules, bounded by depth and total count."""
        seen = set(sections)
        ids = set(sections)

        for _ in range(max_depth):
            next_ids = set()
            for rule_id in ids:
                if (rule := self.rules.get(rule_id)) is not None:
                    related = rule.get('related', [])
                    for r_id in related:
                        if r_id not in seen:
                            next_ids.add(r_id)
                            seen.add(r_id)
                            if len(seen) >= max_total:
                                return seen
            ids = next_ids
            if not ids:
                break

        return seen
