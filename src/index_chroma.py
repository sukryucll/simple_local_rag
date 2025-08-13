# index_chroma.py
# CSV'deki embedding'leri Chroma'ya indeksler. Eski koleksiyonu drop eder.

import os, sys, re
import pandas as pd
from typing import List

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import EMBEDDINGS_CSV, CHROMA_DIR, CHROMA_COLLNAME
import chromadb

def _pick_columns(df: pd.DataFrame) -> tuple[str, List[str]]:
    # Metin sütunu: 'chunk' varsa onu, yoksa 'text'
    chunk_col = "chunk" if "chunk" in df.columns else "text" if "text" in df.columns else None
    if chunk_col is None:
        raise ValueError("CSV içinde ne 'chunk' ne de 'text' sütunu yok.")

    # Sadece adı tamamen rakam olan embedding sütunlarını seç: "0","1",...,"D-1"
    digit_cols = [c for c in df.columns if isinstance(c, str) and re.fullmatch(r"\d+", c)]
    if not digit_cols:
        raise ValueError("Embedding sütunları bulunamadı (adları '0','1',...' şeklinde olmalı).")
    digit_cols = sorted(digit_cols, key=lambda x: int(x))
    return chunk_col, digit_cols

def build_index(batch_size: int = 1000):
    if not os.path.exists(EMBEDDINGS_CSV):
        raise FileNotFoundError(f"Embedding CSV bulunamadı: {EMBEDDINGS_CSV}")

    df = pd.read_csv(EMBEDDINGS_CSV)
    chunk_col, emb_cols = _pick_columns(df)

    docs = df[chunk_col].astype(str).fillna("").tolist()
    embs = df[emb_cols].astype("float32").values.tolist()
    ids  = [str(i) for i in range(len(df))]

    # Metadata (opsiyonel): 'page' varsa ekle
    metas = [{"page": int(p)} if pd.notna(p) else {} for p in df.get("page", pd.Series([None]*len(df)))]

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Eski koleksiyonu tamamen silip yeniden yarat
    try:
        client.delete_collection(name=CHROMA_COLLNAME)
    except Exception:
        pass
    coll = client.create_collection(name=CHROMA_COLLNAME, metadata={"hnsw:space": "cosine"})

    n = len(ids)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        coll.add(ids=ids[s:e], documents=docs[s:e], metadatas=metas[s:e], embeddings=embs[s:e])
        print(f"→ Added {e}/{n}")
    print(f"Indexed {n} chunks into Chroma → {CHROMA_DIR}/{CHROMA_COLLNAME}")

if __name__ == "__main__":
    build_index()
