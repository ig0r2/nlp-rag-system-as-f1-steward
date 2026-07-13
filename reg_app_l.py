import re
import json
from dataclasses import dataclass, field, asdict


@dataclass
class Article:
    number: str  # e.g. "9.1"
    title: str | None  # e.g. "Car Livery"
    text: str  # full text including sub-article content


@dataclass
class Section:
    number: str  # e.g. "9"
    title: str  # e.g. "CAR LIVERY AND COMPETITION NUMBERS"
    articles: list[Article] = field(default_factory=list)


# ── Patterns ──────────────────────────────────────────────────────────────────
RE_SECTION = re.compile(r'^(\d+)\)\s+(.+)$')
RE_ARTICLE = re.compile(r'^(\d+\.\d+)\s+(.*)')
RE_SUB = re.compile(r'^[a-z]\)\s+')

RE_NOISE = re.compile(
    r'^.*ANNEXE L$'
    r'|^APPENDIX L$'
    r'|^Page.*$',
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
    print(text)
    lines = [l.strip() for l in text.splitlines()]

    sections: list[Section] = []
    cur_section: Section | None = None
    cur_article: Article | None = None

    for line in lines:
        if not line:
            continue

        # Section header: "9) CAR LIVERY AND COMPETITION NUMBERS"
        if m := RE_SECTION.match(line):
            cur_section = Section(number=m.group(1), title=m.group(2).strip())
            sections.append(cur_section)
            cur_article = None

        # Article: "9.1 Car Livery" or "8.6 No more than..."
        elif m := RE_ARTICLE.match(line):
            number = m.group(1)
            rest = m.group(2).strip()

            # Detect a standalone title (short, no period at end)
            body = rest
            title = f"ISC Appendix L {number}"

            cur_article = Article(number=number, title=title, text=body)
            if cur_section:
                cur_section.articles.append(cur_article)

        # Sub-article "a) ..." or continuation — append to current article text
        elif cur_article:
            # Add \n before the "a) " / "b) "
            body = RE_SUB.sub('\n' + line, line)
            cur_article.text = (cur_article.text + ' ' + body).strip()

    return sections


# ── Serialise ─────────────────────────────────────────────────────────────────
def to_json(sections: list[Section], **kwargs) -> str:
    return json.dumps([asdict(s) for s in sections], indent=2,
                      ensure_ascii=False, **kwargs)


import pdfplumber


def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            width = page.width
            height = page.height
            right_half_bbox = (width / 2, 0, width, height)

            cropped = page.crop(right_half_bbox)
            text += cropped.extract_text() or ""
        return text


raw = extract_text("data/regulations/appendix_l_2025_publie_le_10_decembre_2025_chapter_4.pdf")
sections = parse_regulations(raw)

# Pretty-print
sections_json = to_json(sections)
print(sections_json)
sections_json = json.loads(sections_json)

# with open('data/regulations/json/appendix-l.json', 'w') as f:
#     json.dump(sections_json, f, indent=2, ensure_ascii=False)

# Or query directly
for sec in sections:
    for art in sec.articles:
        if art.number == "9.1":
            print(art.title)
            print(art.text)

for sec in sections_json:
    for art in sec["articles"]:
        if art["title"] == "SR Article 9.1":
            print(art["text"])
