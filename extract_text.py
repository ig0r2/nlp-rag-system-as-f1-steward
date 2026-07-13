from pathlib import Path
import json
import pdfplumber


def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


FIELDS = [
    {"key": "session", "start": ["Session"], "end": ["Fact", "Infringement"]},
    {"key": "fact", "start": ["Fact"], "end": ["Infringement", "Infringment", "Offence"]},
    {"key": "infringement", "start": ["Infringement", "Infringment", "Offence"], "end": ["Decision"]},
    {"key": "decision", "start": ["Decision"], "end": ["Reason"]},
    {"key": "reason", "start": ["Reason"],
     "end": ["Competitors are reminded", "Gerd Ennser", "Garry Connelly", "Nish Shetty"]},
]


def extract_fields(text):
    results = {}
    for field in FIELDS:
        # check for start
        start_idx = -1
        for start_str in field["start"]:
            start_idx = text.find(start_str)
            if start_idx == -1:
                continue
            start_idx += len(start_str)
            break
        if start_idx == -1:
            results[str(field["key"])] = ""
            continue

        # check for end
        end_idx = 0
        for end_str in field["end"]:
            end_idx = text[start_idx:].find(end_str)
            if end_idx == -1:
                results[str(field["key"])] = ""
                continue
            break

        results[str(field["key"])] = text[start_idx: start_idx + end_idx].strip()
    return results


if __name__ == "__main__":
    # Group PDFs by year — pdf/[year]/[race]/*.pdf
    # Structure: pdf/<year>/<race>/<file>.pdf
    race_incidents: dict[str, list[dict]] = {}

    for pdf_path in sorted(Path("data/pdf").glob("**/*.pdf")):
        parts = pdf_path.parts  # ('data', 'pdf', year, race, filename)
        if len(parts) < 5:
            print(f"Skipping unexpected path structure: {pdf_path}")
            continue

        year = parts[2]

        text = extract_text(pdf_path)
        fields = extract_fields(text)

        if text.find("Title") != -1:
            print("WARNING: Delete page 1 in ", pdf_path)

        for name, value in fields.items():
            if value == "":
                print(f"WARNING: empty field '{name}' in {pdf_path}")

        incident: dict = {
            "filename": pdf_path.name,
            "year": year,
            "session": fields["session"].replace("\n", " "),
            "infringement": fields["infringement"].replace("\n", " "),
            "decision": fields["decision"].replace("\n", " "),
            "fact": fields["fact"].replace("\n", " "),
            "reason": fields["reason"].replace("\n", " "),
        }

        race_incidents.setdefault(year, []).append(incident)

    # Write one JSON file per race: json/[year]/race.json
    for year, incidents in race_incidents.items():
        out_dir = Path("data/json_raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{year}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(incidents, f, ensure_ascii=False, indent=2)
        print(f"Written {len(incidents)} incidents → {out_file}")

    print("Done.")
