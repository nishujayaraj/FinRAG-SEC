import os
import re
from pathlib import Path
from bs4 import BeautifulSoup


def parse_html_filing(filepath: str) -> dict:
    """Parse a raw SEC HTML filing into clean text."""
    
    # Open and read the raw HTML file from disk
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()

    # BeautifulSoup parses the HTML structure
    soup = BeautifulSoup(html, "html.parser")

    # Remove all junk tags — scripts, styles, hidden elements
    for tag in soup(["script", "style", "meta", "noscript"]):
        tag.decompose()

    # Also remove XBRL/XML data sections — these are machine-readable
    # accounting codes, not human text. They look like <ix:header> tags
    for tag in soup(["ix:header", "ix:hidden", "ix:nonnumeric"]):
        tag.decompose()

    # Remove any tag that contains a URL pattern like fasb.org or xbrl
    # These are accounting taxonomy references, not useful text
    for tag in soup.find_all(True):
        text_content = tag.get_text()
        if "fasb.org" in text_content[:100] or "xbrl" in text_content[:100].lower():
            if len(text_content) < 500:  # only remove small tags, not whole sections
                tag.decompose()

    # Extract visible text
    text = soup.get_text(separator="\n")

    # Clean up line by line
    lines = []
    for line in text.splitlines():
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip lines that look like URLs or XML paths
        if line.startswith("http") or line.startswith("//"):
            continue
        
        # Skip lines that are just accounting codes like "P3Y" or "FY"
        if re.match(r'^[A-Z0-9#]{1,10}$', line):
            continue
        
        # Skip lines with only numbers or special characters
        if re.match(r'^[\d\s\.\,\-\$\%]+$', line) and len(line) < 20:
            continue
            
        lines.append(line)

    text = "\n".join(lines)

    # Remove long lines of dashes/underscores used as dividers in SEC filings
    text = re.sub(r"[-_=]{5,}", "", text)
    
    # Collapse multiple blank lines into max 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Extract company and date from filename
    # e.g. "apple_2024-11-01_10K.html" → company="apple", date="2024-11-01"
    filename = Path(filepath).stem
    parts = filename.split("_")
    company = parts[0]
    date = parts[1]

    return {
        "company": company,
        "date": date,
        "filepath": filepath,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split())
    }


def parse_all_filings(raw_dir: str = "data/raw",
                      processed_dir: str = "data/processed") -> list:
    """Parse all HTML filings in raw_dir and return list of parsed docs."""

    # Make sure the processed folder exists, create it if not
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    parsed_docs = []

    # Find all .html files inside data/raw/
    html_files = list(Path(raw_dir).glob("*.html"))

    print(f"Found {len(html_files)} filings to parse...\n")

    # Loop through each file and parse it one by one
    for filepath in sorted(html_files):
        print(f"Parsing {filepath.name}...")
        doc = parse_html_filing(str(filepath))
        parsed_docs.append(doc)
        print(f"  ✅ {doc['company']} | {doc['date']} | "
              f"{doc['word_count']:,} words | {doc['char_count']:,} chars")

    print(f"\n✅ Parsed {len(parsed_docs)} filings total.")
    return parsed_docs


if __name__ == "__main__":
    # Parse all 15 filings
    docs = parse_all_filings()

    # Print first 1000 characters of the first document
    # so that we can visually verify the text looks clean
    print("\n--- SAMPLE TEXT FROM FIRST DOCUMENT ---")
    print(docs[0]["text"][:1000])