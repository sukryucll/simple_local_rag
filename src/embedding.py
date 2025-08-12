# embedding.py
import os, sys, math
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import HF_MODEL, E5_USE_PREFIXES
from config import CHUNKS_CSV, EMBEDDINGS_CSV

def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1)               # [B,L,1]
    summed = (last_hidden_state * mask).sum(dim=1)    # [B,D]
    lengths = mask.sum(dim=1).clamp_min(1)            # [B,1]
    return summed / lengths                           # [B,D]

@torch.inference_mode()
def embed_chunks(batch_size: int = 64):
    if not os.path.exists(CHUNKS_CSV):
        raise FileNotFoundError(f"Missing {CHUNKS_CSV}")

    df = pd.read_csv(CHUNKS_CSV)
    texts = df["chunk"].fillna(" ").astype(str).tolist()

    if E5_USE_PREFIXES:
        # E5 önerisi (isteğe bağlı): "passage: "
        texts = ["passage: " + t for t in texts]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    mdl = AutoModel.from_pretrained(HF_MODEL).to(device).eval()

    embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch = tok(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = mdl(**batch).last_hidden_state           # [B,L,D]
        pooled = _mean_pool(out, batch["attention_mask"]).cpu()  # [B,D]
        embs.append(pooled)

        print(f"→ embedded {min(i+batch_size, len(texts))}/{len(texts)}", end="\r")

    embs = torch.cat(embs, dim=0).numpy()             # [N,D]
    emb_df = pd.DataFrame(embs)                       # columns: 0..D-1
    out_df = pd.concat([df[["page", "chunk"]].reset_index(drop=True), emb_df], axis=1)

    os.makedirs(os.path.dirname(EMBEDDINGS_CSV), exist_ok=True)
    out_df.to_csv(EMBEDDINGS_CSV, index=False, encoding="utf-8")
    print(f"\n✅ {len(texts)} chunks embedded → {EMBEDDINGS_CSV}")
    return out_df

if __name__ == "__main__":
    embed_chunks()
