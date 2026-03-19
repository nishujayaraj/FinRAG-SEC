import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# SEC requires a User-Agent header to identify who is making requests
HEADERS = {
    "User-Agent": "FinRAG-SEC nischithasjayaraj@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# Some well-known companies and their SEC CIK numbers
COMPANY_CIK = {
    "apple": "0000320193",
    "microsoft": "0000789019",
    "tesla": "0001318605",
    "google": "0001652044",
    "amazon": "0001018724",
}

def get_10k_filings(company_name: str, max_filings: int = 3) -> list:
    """Fetch the last N 10-K filing URLs for a company."""
    cik = COMPANY_CIK.get(company_name.lower())
    if not cik:
        raise ValueError(f"Company '{company_name}' not found. Choose from: {list(COMPANY_CIK.keys())}")

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    filings = data["filings"]["recent"]
    forms = filings["form"]
    accession_numbers = filings["accessionNumber"]
    filing_dates = filings["filingDate"]
    primary_docs = filings["primaryDocument"]

    results = []
    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accession_numbers[i].replace("-", "")
            doc = primary_docs[i]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            results.append({
                "company": company_name,
                "date": filing_dates[i],
                "url": filing_url,
                "accession": accession_numbers[i]
            })
        if len(results) >= max_filings:
            break

    return results


def download_filing(filing: dict, save_dir: str = "data/raw") -> str:
    """Download a single filing and save it locally."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{filing['company']}_{filing['date']}_10K.html"
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath):
        print(f"Already exists: {filepath}")
        return filepath

    print(f"Downloading {filing['company']} 10-K from {filing['date']}...")
    response = requests.get(filing["url"], headers=HEADERS)
    response.raise_for_status()

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Saved to {filepath}")
    time.sleep(0.5)  # Be polite to SEC servers
    return filepath


if __name__ == "__main__":
    companies = ["apple", "microsoft", "tesla", "google", "amazon"]
    
    all_filings = []
    for company in companies:
        print(f"\nFetching {company.upper()} filings...")
        filings = get_10k_filings(company, max_filings=3)
        for filing in filings:
            print(f"  Found: {filing['date']} → {filing['url']}")
            download_filing(filing)
            all_filings.append(filing)
    
    print(f"\n✅ Done! Downloaded {len(all_filings)} filings total.")