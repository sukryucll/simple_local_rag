# main.py (root)

import os
import sys
import pandas as pd

# 1) src klasörünü path'e ekle
ROOT    = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 2) Ayarları ve modülleri yükle
from config import (
    PAGES_CSV, CHUNKS_CSV, EMBEDDINGS_CSV,
    USE_RERANKER, CANDIDATE_K, W_CE, W_SIM, W_LEX
)
from ingestion import load_pdf_pages
from chunking import chunk_pages

# embed_chunks ismini kullan; eski dosyanda embed_chunks_manual varsa ona düş
try:
    from embedding import embed_chunks
except ImportError:
    from embedding import embed_chunks_manual as embed_chunks

from index_chroma import build_index
from retrieval_chroma import retrieve_top_k, _get_collection
from prompting import format_prompt, ask_llm

# Opsiyonel: reranker (yoksa sorun olmasın)
try:
    from search_rerank import search_and_rerank
except Exception:
    search_and_rerank = None


def prepare_data():
    """
    PDF -> pages.csv -> chunks.csv -> embeddings.csv
    Ardından Chroma indexini (boşsa veya yeni embedding üretildiyse) kurar.
    """
    # PDF → pages.csv
    if not os.path.exists(PAGES_CSV):
        pages = load_pdf_pages()
        os.makedirs(os.path.dirname(PAGES_CSV), exist_ok=True)
        pd.DataFrame({"text": pages}).to_csv(PAGES_CSV, index=False, encoding="utf-8")
    else:
        pages = pd.read_csv(PAGES_CSV)["text"].tolist()

    # pages.csv → chunks.csv
    need_embed = False
    if not os.path.exists(CHUNKS_CSV):
        chunks = chunk_pages(pages)
        os.makedirs(os.path.dirname(CHUNKS_CSV), exist_ok=True)
        pd.DataFrame(chunks).to_csv(CHUNKS_CSV, index=False, encoding="utf-8")
        need_embed = True

    # chunks.csv → embeddings.csv
    need_reindex = False
    if not os.path.exists(EMBEDDINGS_CSV) or need_embed:
        embed_chunks()          # EMBEDDINGS_CSV dosyasını üretir
        need_reindex = True

    # Chroma index (boşsa veya yeni embedding üretildiyse)
    try:
        coll = _get_collection()
        is_empty = (coll.count() == 0)
    except Exception:
        is_empty = True

    if need_reindex or is_empty:
        build_index()


def _score_of(hit):
    return hit.get("fused_score", hit.get("score"))

def run(query: str, top_k: int = 5):
    # 0) Gerekli verileri hazırla (embed + index gerekirse)
    prepare_data()

    # 0.1) Opsiyonel eşik ve A/B görünürlüğü (config'te yoksa varsayılan kullan)
    try:
        from config import RERANK_GATE_THRESHOLD, VERBOSE_AB
    except Exception:
        RERANK_GATE_THRESHOLD = 0.85  # base top1 skoru bunun altındaysa reranker tetikler
        VERBOSE_AB = False            # True yaparsan mini A/B çıktısı basar

    # 1) Önce saf Chroma ile getir (biraz geniş tut ki gerekirse kıyaslayalım)
    candidate_k = max(top_k, CANDIDATE_K) if isinstance(CANDIDATE_K, int) else top_k
    base_hits = retrieve_top_k(query, k=candidate_k)
    if not base_hits:
        print("\nNo results from retrieval.")
        return

    base_top1 = base_hits[0]
    base_score = base_top1.get("score") or 0.0

    # 2) Reranker kullanılsın mı? (flag + gating)
    used_reranker = False
    hits = base_hits[:top_k]

    if USE_RERANKER and (search_and_rerank is not None):
        gate = (base_score < float(RERANK_GATE_THRESHOLD))
        if gate:
            rer = search_and_rerank(
                query,
                candidate_k=candidate_k,
                top_k=top_k,
                w_ce=W_CE, w_sim=W_SIM, w_lex=W_LEX
            )
            if rer:  # güvenli
                hits = rer
                used_reranker = True

    # 3) Retrieved contexts (kısa özet)
    def _score_of(h): return h.get("fused_score", h.get("score"))
    print("\n=== Retrieved Contexts ===")
    hdr = f"(reranker={'ON' if used_reranker else 'OFF'}; base_top1={base_score:.4f}; gate<{RERANK_GATE_THRESHOLD:.2f})"
    print(hdr)
    for i, h in enumerate(hits, 1):
        page = h.get("page")
        score = _score_of(h)
        page_str = str(page) if page is not None else "-"
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "NA"
        print(f"\n#{i} - Page {page_str}, score {score_str}\n{h['chunk']}")

    # 3.1) (Opsiyonel) mini A/B snapshot: base top1 vs final top1
    if VERBOSE_AB:
        import re
        STOP = set("the a an and or of to in on for with by from at as is are was were be been being this that these those".split())
        def lex_hits(q, text):
            toks = [t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z\\-]+", q)]
            toks = [t for t in toks if t not in STOP and len(t) >= 4]
            tl = (text or "").lower()
            return sum(tl.count(t) for t in toks)
        def snippet(t, n=160): return (t or "").replace("\n", " ").strip()[:n]

        final_top1 = hits[0]
        blex = lex_hits(query, base_top1.get("chunk", ""))
        flex = lex_hits(query, final_top1.get("chunk", ""))
        changed = snippet(base_top1.get("chunk", ""), 80) != snippet(final_top1.get("chunk", ""), 80)
        print("\n--- A/B snapshot ---")
        print(f" Base | page={base_top1.get('page')} | score={base_score:.4f} | lex={blex} | {snippet(base_top1.get('chunk',''))}...")
        fscore = _score_of(final_top1)
        print(f" Final| page={final_top1.get('page')} | score={fscore:.4f} | lex={flex} | {snippet(final_top1.get('chunk',''))}...")
        print(" Result:", "↑ improved" if flex > blex else "= no gain", "|", "changed" if changed else "same")

    # 4) Prompt & LLM cevabı
    contexts = [h["chunk"] for h in hits]
    pages    = [h.get("page") for h in hits]
    prompt   = format_prompt(query, contexts)
    answer   = ask_llm(prompt)

    # 5) Cevabı göster
    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    q = input("Your question: ").strip()
    if q:
        run(q)
