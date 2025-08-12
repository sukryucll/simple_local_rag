# search_rerank.py
# Chroma'dan geniş aday kümesi alır, CrossEncoder + vektör benzerliği + anahtar kelime eşleşmesi ile
# yeniden sıralar (fusion reranking).

import os, sys, re
import numpy as np
from typing import List, Tuple
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sentence_transformers import CrossEncoder
from retrieval_chroma import _get_collection, encode_query  # mevcut modülden
# Not: encode_query ile aynı encoder'ı kullandığımız garanti.

# ---- CrossEncoder tek sefer yükle ----
_CE = None
def _load_ce():
    global _CE
    if _CE is None:
        # MS MARCO için hafif ve iyi bir reranker
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _CE

# ---- Yardımcılar ----
def _extract_keywords(q: str) -> List[str]:
    ql = q.lower()
    base = {w for w in re.split(r"\W+", ql) if len(w) >= 4}
    # Basit genişletmeler (örnek): brush-border & peptidase terminolojisi
    if "brush" in base or "border" in base or "brush-border" in ql or "brush border" in ql:
        base.update(["brush-border", "brush border", "microvilli", "microvillus"])
    if "peptidase" in base or "peptidases" in ql or "peptididase" in ql:
        base.update(["peptidase", "peptidases", "aminopeptidase", "dipeptidase", "exopeptidase"])
    if "protein" in base and "digestion" in base:
        base.update(["enterocyte", "small intestine", "intestinal"])
    return list(base)

def _lex_score(doc: str, kws: List[str]) -> float:
    t = doc.lower()
    s = 0.0
    for kw in kws:
        if not kw: 
            continue
        c = t.count(kw)
        if c > 0:
            s += np.log1p(c)  # azalan marjinal katkı
    return float(s)

def _z(x: List[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return a
    mu, sd = a.mean(), a.std()
    if sd == 0:
        return np.zeros_like(a)
    return (a - mu) / sd

# ---- Ana arama + rerank ----
def search_and_rerank(query: str, candidate_k: int = 100, top_k: int = 5,
                      w_ce: float = 1.0, w_lex: float = 0.5, w_sim: float = 0.4):
    coll = _get_collection()
    # Chroma: embedding ile geniş aday topla (mesafeyi da iste)
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

    # 1) CrossEncoder skoru
    ce = _load_ce()
    pairs = [(query, d) for d in docs]
    ce_scores = ce.predict(pairs)  # rakamsal; işaret/ölçek önemli değil, z-normalize edeceğiz

    # 2) Vektör benzerliği (cosine similarity ~ 1 - distance)
    sim_scores = [1.0 - float(d) if d is not None else 0.0 for d in dists]

    # 3) Anahtar kelime eşleşmesi
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
        results.append({
            "rank": rank,
            "page": page,
            "fused_score": float(fused[idx]),
            "ce": float(ce_scores[idx]),
            "sim": float(sim_scores[idx]),
            "lex": float(lex_scores[idx]),
            "chunk": docs[idx],
        })
    return results

# ---- CLI ----
if __name__ == "__main__":
    try:
        while True:
            q = input("Soru: ").strip()
            if not q: break
            hits = search_and_rerank(q, candidate_k=100, top_k=5)
            for h in hits:
                print(f"\n# {h['rank']} | page {h['page']} | fused {h['fused_score']:.3f} "
                      f"(ce {h['ce']:.3f} / sim {h['sim']:.3f} / lex {h['lex']:.3f})\n{h['chunk']}")
    except (KeyboardInterrupt, EOFError):
        pass
