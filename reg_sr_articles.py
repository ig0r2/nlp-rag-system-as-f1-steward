import re
import json
from dataclasses import dataclass, field, asdict


@dataclass
class Article:
    id: str
    title: str | None  # e.g. "Car Livery"
    text: str  # full text including sub-article content


@dataclass
class Section:
    id: str
    title: str  # e.g. "CAR LIVERY AND COMPETITION NUMBERS"


# ── Patterns ──────────────────────────────────────────────────────────────────
RE_SECTION = re.compile(r'^(\d+)\)\s+(.+)$')
RE_ARTICLE = re.compile(r'^(\d+\.\d+)\s+(.*)')
RE_SUB = re.compile(r'^[a-z]\)\s+')

RE_NOISE = re.compile(
    r'^\d{4} Formula 1 Sporting Regulations.*$'
    r'|^©\d{4} Fédération.*$'
    r'|^\d+/\d+\s*\d+\s+\w+.*$',
    re.MULTILINE
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = RE_NOISE.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_structural(line: str) -> bool:
    """True if the line starts a new section or article (not sub-article)."""
    return bool(RE_SECTION.match(line) or RE_ARTICLE.match(line))


# ── Main parser ───────────────────────────────────────────────────────────────
def parse_regulations(raw: str) -> list[Section]:
    text = clean(raw)
    lines = [l.strip() for l in text.splitlines()]

    sections: list[Section] = []
    cur_section: Section | None = None
    cur_article: Article | None = None

    for line in lines:
        if not line:
            continue

        # Section header: "9) CAR LIVERY AND COMPETITION NUMBERS"
        if m := RE_SECTION.match(line):
            id = f"SR Article {m.group(1)}"
            cur_section = Section(title=m.group(2).strip(), id=id)
            sections.append(cur_section)
            cur_article = None

        # Article: "9.1 Car Livery" or "8.6 No more than..."
        elif m := RE_ARTICLE.match(line):
            rest = m.group(2).strip()

            # Detect a standalone title (short, no period at end)
            if rest and rest[0].isupper() and len(rest) < 50 and not rest.endswith('.'):
                title, body = rest, ''
            else:
                title, body = "", rest

            id = f"SR Article {m.group(1)}"

            cur_article = Article(title=title, text=body, id=id)
            sections.append(cur_article)

        # Sub-article "a) ..." or continuation — append to current article text
        elif cur_article:
            # Add \n before the "a) " / "b) "
            body = RE_SUB.sub('\n' + line, line)
            cur_article.text = (cur_article.text + ' ' + body).strip()

    return sections


def to_json(sections: list[Section], **kwargs) -> str:
    return json.dumps([asdict(s) for s in sections], indent=2,
                      ensure_ascii=False, **kwargs)


def to_dict(sections_json):
    return {section["id"]: {k: v for k, v in section.items() if k != "id"}
            for section in sections_json}


import pdfplumber


def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text


raw = extract_text("data/regulations/pdf/FIA 2025 Formula 1 Sporting Regulations - Issue 5 - 2025-04-30 - Articles.pdf")
sections = parse_regulations(raw)

# Pretty-print
sections_json = to_json(sections)
print(sections_json)
sections_json = json.loads(sections_json)
sections_json = to_dict(sections_json)

with open('data/regulations/json/sr-articles.json', 'w', encoding='utf-8') as f:
    json.dump(sections_json, f, indent=2, ensure_ascii=False)

# Or query directly
for sec in sections:
    if sec.id == "SR Article 9.1":
        print(sec.title)
        print(sec.text)

# for sec in sections_json:
#     if sec["id"] == "SR Article 9.1":
#         print(sec["title"])
#         print(sec["text"])

sec = sections_json["SR Article 9.1"]
print(sec["title"])
print(sec["text"])
