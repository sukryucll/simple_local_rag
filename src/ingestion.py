# src/ingestion.py
import os, sys
import fitz  
import pandas as pd

# Proje kökünü ekle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import PDF_PATH, PAGES_CSV

def load_pdf_pages(pdf_path=PDF_PATH):
    pages = []
    doc = fitz.open(pdf_path)
    for page in doc:
        cleaned = []
        for line in page.get_text().splitlines():
            low = line.lower()
            # Link ve kısa başlık satırlarını at
            if low.startswith("http") or "view it online" in low:
                continue
            if len(line.split()) < 5:
                continue
            cleaned.append(line)
        pages.append("\n".join(cleaned))
    return pages

if __name__ == "__main__":
    pages = load_pdf_pages()
    os.makedirs(os.path.dirname(PAGES_CSV), exist_ok=True)
    pd.DataFrame({"text": pages}).to_csv(PAGES_CSV, index=False, encoding="utf-8")
    print(f"Extracted {len(pages)} cleaned pages → {PAGES_CSV}")