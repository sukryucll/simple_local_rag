# src/chunking.py
import os, sys
import pandas as pd

# Proje kökünü ekle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import PAGES_CSV, CHUNKS_CSV, MAX_CHARS, OVERLAP

def chunk_text(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks

def chunk_pages(pages):
    all_chunks = []
    for idx, text in enumerate(pages, 1):
        for c in chunk_text(text):
            all_chunks.append({"page": idx, "chunk": c})
    return all_chunks

if __name__ == "__main__":
    df = pd.read_csv(PAGES_CSV)
    texts = df["text"].fillna("").astype(str).tolist()
    raw_chunks = chunk_pages(texts)
    # Gereksiz kısa parçaları el: en az 20 kelime ve en az 2 satırlı
    filtered = [c for c in raw_chunks 
                if len(c['chunk'].split()) >= 20 and c['chunk'].count("\n") >= 2]
    os.makedirs(os.path.dirname(CHUNKS_CSV), exist_ok=True)
    pd.DataFrame(filtered).to_csv(CHUNKS_CSV, index=False, encoding="utf-8")
    print(f"{len(filtered)} chunks → {CHUNKS_CSV}")