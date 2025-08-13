# search_rerank.py
# Chroma'dan geniş aday kümesi alır, CrossEncoder + vektör benzerliği + BM25 leksik sinyali ile
# yeniden sıralar (fusion reranking).

import os, sys, re, math
import numpy as np
from collections import Counter
from typing import List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sentence_transformers import CrossEncoder
from retrieval_chroma import _get_collection, encode_query  

# CrossEncoder tek sefer yükle
_CE = None
def _load_ce():
    global _CE
    if _CE is None:
        # MS MARCO için hafif ve iyi bir reranker
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _CE


# BM25 yardımcıları (genel)

# Basit bir stopword listesi (genel amaçlı)
_STOP = {
    "the","a","an","and","or","but","if","then","else","of","to","in","on","for","with","as",
    "is","are","was","were","be","being","been","by","at","from","it","that","this","these","those",
    "i","you","he","she","we","they","them","his","her","their","our","your","my","me","us"
}

def _tokens(text: str) -> List[str]:
    """Küçük harf + tireleri boşluğa çevir + alfa-nümerik olmayanlara göre böl + kısa ve stopword'leri at."""
    t = text.lower().replace("-", " ")
    return [w for w in re.split(r"\W+", t) if len(w) >= 2 and w not in _STOP]

# BM25 global parametreleri (candidate havuzu üstünden hazırlanır)
_BM25_K1: float = 1.5
_BM25_B: float  = 0.75
_BM25_IDF: dict = {}
_BM25_AVGDL: float = 0.0

def _prepare_bm25(docs: List[str]) -> None:
    """Aday dokümanlar üzerinde lokal BM25 parametrelerini hazırla (IDF ve ort. doküman uzunluğu)."""
    global _BM25_IDF, _BM25_AVGDL
    docs_toks = [_tokens(d) for d in docs]
    N = len(docs_toks)
    if N == 0:
        _BM25_IDF, _BM25_AVGDL = {}, 0.0
        return
    _BM25_AVGDL = sum(len(d) for d in docs_toks) / N

    df = Counter()
    for d in docs_toks:
        df.update(set(d))
    # Güvenli IDF (BM25+ tarzı): log(1 + (N - df + 0.5)/(df + 0.5))
    _BM25_IDF = {t: math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5)) for t, df_t in df.items()}

# Yardımcılar (mevcut imzalar korunur)
def _extract_keywords(q: str) -> List[str]:
    """
    GENEL anahtar terim çıkarımı (BM25 için tokenlar).
    - domain-özel genişletme YOK; tamamen genel çalışır.
    """
    return _tokens(q)

def _lex_score(doc: str, kws: List[str]) -> float:
    """
    BM25 skoru (lokal, candidate_havuzu üzerinde hazırlanan IDF ile).
    Buradaki 'kws' = sorgu tokenları.
    """
    if not _BM25_IDF or _BM25_AVGDL <= 0:
        # Güvenlik: prepare_bm25 çağrılmadıysa sıfır dön.
        return 0.0

    toks = _tokens(doc)
    tf = Counter(toks)
    dl = len(toks)
    denom_norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (dl / _BM25_AVGDL))

    s = 0.0
    # Sorgu tarafında çoğulları iki kez saymamak için set() kullanılır (standart BM25 pratiği)
    for t in set(kws):
        f = tf.get(t, 0)
        if f == 0:
            continue
        idf = _BM25_IDF.get(t, 0.0)
        s += idf * (f * (_BM25_K1 + 1.0)) / (f + denom_norm)
    return float(s)

def _z(x: List[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return a
    mu, sd = a.mean(), a.std()
    if sd == 0:
        return np.zeros_like(a)
    return (a - mu) / sd


# Ana arama + rerank 
def search_and_rerank(query: str, candidate_k: int = 100, top_k: int = 5,
                      w_ce: float = 1.0, w_lex: float = 0.5, w_sim: float = 0.4):
    coll = _get_collection()
    # Chroma: embedding ile geniş aday topla (mesafeyi de iste)
    q_emb = encode_query(query)
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"]
    )
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    if not docs:
        return []

    # BM25 parametrelerini aday havuzu üzerinde hazırla
    _prepare_bm25(docs)

    # 1) CrossEncoder skoru
    ce = _load_ce()
    pairs = [(query, d) for d in docs]
    ce_scores = ce.predict(pairs)  
    
    # 2) Vektör benzerliği (cosine similarity ~ 1 - distance)
    sim_scores = [1.0 - float(d) if d is not None else 0.0 for d in dists]

    # 3) BM25 leksik skoru
    kws = _extract_keywords(query)            
    lex_scores = [_lex_score(d, kws) for d in docs]

    # Z-normalize ve ağırlıklı füzyon
    z_ce  = _z(ce_scores)
    z_sim = _z(sim_scores)
    z_lex = _z(lex_scores)
    fused = w_ce * z_ce + w_sim * z_sim + w_lex * z_lex

    order = np.argsort(-fused)[:top_k]  # büyükten küçüğe
    results = []
    for rank, idx in enumerate(order, 1):
        meta = metas[idx] if idx < len(metas) else {}
        page = meta.get("page") if isinstance(meta, dict) else None
        bm25_val = float(lex_scores[idx])
        results.append({
            "rank": rank,
            "page": page,
            "fused_score": float(fused[idx]),
            "ce": float(ce_scores[idx]),
            "sim": float(sim_scores[idx]),
            "bm25": bm25_val,         # <-- yeni anahtar (tercih edilen)
            "lex": bm25_val,          # <-- geriye uyumlu alias
            "chunk": docs[idx],
        })
    return results

if __name__ == "__main__":
    try:
        while True:
            q = input("Soru: ").strip()
            if not q: break
            hits = search_and_rerank(q, candidate_k=100, top_k=5)
            for h in hits:
                print(f"\n# {h['rank']} | page {h['page']} | fused {h['fused_score']:.3f} "
                      f"(ce {h['ce']:.3f} / sim {h['sim']:.3f} / bm25 {h['bm25']:.3f})\n{h['chunk']}")
    except (KeyboardInterrupt, EOFError):
        pass
