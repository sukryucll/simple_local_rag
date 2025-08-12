# retrieval_chroma.py
import os, sys, torch
import chromadb
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import HF_MODEL, E5_USE_PREFIXES
from config import CHROMA_DIR, CHROMA_COLLNAME

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_encoder():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        _model = AutoModel.from_pretrained(HF_MODEL).to(_device).eval()
    return _tokenizer, _model

def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp_min(1)
    return summed / lengths

@torch.inference_mode()
def encode_query(text: str):
    # E5 önerisi (isteğe bağlı): "query: "
    if E5_USE_PREFIXES:
        text = "query: " + text
    tok, mdl = _load_encoder()
    batch = tok([text], padding=True, truncation=True, return_tensors="pt").to(_device)
    out = mdl(**batch).last_hidden_state
    emb = _mean_pool(out, batch["attention_mask"])
    return emb.squeeze(0).detach().cpu().tolist()

_client = None
_coll = None
def _get_collection():
    global _client, _coll
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _coll = _client.get_or_create_collection(
            name=CHROMA_COLLNAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _coll

def retrieve_top_k(query: str, k: int = 5):
    coll = _get_collection()
    if coll.count() == 0:
        raise RuntimeError("Chroma collection is empty. Run: python index_chroma.py")

    q_emb = encode_query(query)
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids   = res.get("ids", [[str(i) for i in range(len(docs))]])[0]

    results = []
    for i, doc in enumerate(docs):
        page = metas[i].get("page") if i < len(metas) and isinstance(metas[i], dict) else None
        dist = float(dists[i]) if i < len(dists) else None
        score = (1.0 - dist) if dist is not None else None
        results.append({"rank": i+1, "id": ids[i], "page": page, "distance": dist, "score": score, "chunk": doc})
    return results

if __name__ == "__main__":
    try:
        while True:
            q = input("Soru: ").strip()
            if not q: break
            hits = retrieve_top_k(q, k=5)
            for h in hits:
                s = f"{h['score']:.4f}" if h["score"] is not None else "NA"
                print(f"\n# {h['rank']} | page {h['page']} | score {s}\n{h['chunk']}")
    except (KeyboardInterrupt, EOFError):
        pass
