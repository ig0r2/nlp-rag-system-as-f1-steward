import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

from utils.path import get_project_root

DEFAULT_KEYWORDS = ["infringement", "decision", "offence"]
# DEFAULT_KEYWORDS = ["offence"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_page(url: str) -> BeautifulSoup:
    """Fetch the page and return a BeautifulSoup object."""
    print(f"Fetching page: {url}")
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_pdf_links(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """
    Extract all PDF links from the page.
    Returns a list of dicts with 'url' and 'filename'.
    """
    pdf_links = []
    seen_urls = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()

        # Resolve relative URLs
        full_url = urljoin(base_url, href)

        # Skip duplicates
        if full_url in seen_urls:
            continue

        # Check if it's a PDF link (by extension or content-type hint)
        parsed_path = unquote(urlparse(full_url).path)
        if not parsed_path.lower().endswith(".pdf"):
            continue

        filename = Path(parsed_path).name
        seen_urls.add(full_url)
        pdf_links.append({"url": full_url, "filename": filename})

    return pdf_links


def filename_matches(filename: str, keywords: list[str]) -> bool:
    """Return True if the filename contains any of the keywords (case-insensitive)."""
    name_lower = filename.lower()
    return any(kw.lower() in name_lower for kw in keywords)


def sanitize_filename(filename: str) -> str:
    """Remove characters that are invalid in filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def download_pdf(url: str, dest_path: Path, session: requests.Session) -> bool:
    """
    Download a single PDF to dest_path.
    Returns True on success, False on failure.
    """
    try:
        response = session.get(url, headers=HEADERS, timeout=60, stream=True)
        response.raise_for_status()

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = dest_path.stat().st_size / 1024
        print(f"  ✓ Saved: {dest_path.name}  ({size_kb:.1f} KB)")
        return True

    except requests.RequestException as e:
        print(f"  ✗ Failed to download {url}: {e}")
        return False


def scrape(url: str, output_dir: Path, keywords: list[str], delay: float = 0.5):
    """Main scraping routine."""
    session = requests.Session()

    # 1. Fetch the page
    try:
        soup = fetch_page(url)
    except requests.RequestException as e:
        print(f"Error fetching page: {e}")
        sys.exit(1)

    # 2. Extract all PDF links
    all_pdfs = extract_pdf_links(soup, url)
    print(f"\nFound {len(all_pdfs)} PDF link(s) on the page.")

    # 3. Filter by keywords
    matched = [
        pdf for pdf in all_pdfs if filename_matches(pdf["filename"], keywords)
    ]

    if not matched:
        print(
            f"No PDFs matched keywords: {keywords}\n"
            "All PDF filenames found on the page:"
        )
        for pdf in all_pdfs:
            print(f"  - {pdf['filename']}")
        return

    print(
        f"Matched {len(matched)} PDF(s) containing "
        f"{' / '.join(repr(k) for k in keywords)} in filename:\n"
    )
    for pdf in matched:
        print(f"  • {pdf['filename']}")

    # 4. Download matched PDFs
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading to: {output_dir.resolve()}\n")

    success, failed = 0, 0
    for pdf in matched:
        safe_name = sanitize_filename(pdf["filename"])
        dest = output_dir / safe_name

        # Skip if already downloaded
        if dest.exists():
            print(f"  – Skipping (already exists): {safe_name}")
            success += 1
            continue

        ok = download_pdf(pdf["url"], dest, session)
        if ok:
            success += 1
        else:
            failed += 1

        time.sleep(delay)  # Be polite to the server

    # 5. Summary
    print(f"\nDone. Downloaded: {success}  |  Failed: {failed}")


if __name__ == "__main__":
    """
    let sel = document.querySelector("#facetapi_select_facet_form_2")
    let res = "URLS = ["
    for (let i = 0; i < sel.childElementCount; i++) 
    { res += "'https://fia.com" + sel.children[i].value + "'," }
    res += "]"
    console.log(res)
    """
    # + add current page url

    URLS = [
        'https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/event/Australian%20Grand%20Prix/season/season-2023-2042']

    for url in URLS:
        scrape(url=url, output_dir=get_project_root() / "downloads", keywords=DEFAULT_KEYWORDS)
