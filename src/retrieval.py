import os, sys

# Proje kök dizinini bul ve path'e ekle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import EMBEDDINGS_CSV, HF_MODEL
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def load_embeddings(emb_csv=EMBEDDINGS_CSV):
    # Embedding CSV'den DataFrame ve numpy dizisini döndür
    df = pd.read_csv(emb_csv)
    emb_cols = [c for c in df.columns if c not in ['page', 'chunk']]
    embs = df[emb_cols].values
    return df, embs


def retrieve_top_k(query, df, embs, model_name=HF_MODEL, k=5):
    # Sorguyu embed et
    embedder = SentenceTransformer(model_name)
    q_emb = embedder.encode([query], convert_to_tensor=True)

    # Cosine similarity
    emb_tensor = torch.tensor(embs)
    sims = torch.nn.functional.cosine_similarity(q_emb, emb_tensor, dim=1).numpy()

    # En yüksek skorlu k index
    topk_idx = np.argsort(sims)[::-1][:k]
    return df.iloc[topk_idx].reset_index(drop=True), sims[topk_idx]


if __name__ == "__main__":
    # 1) Embedding veri setini yükle
    df, embs = load_embeddings()

    # 2) Kullanıcıdan soru al
    query = input("Soru: ")

    # 3) İlk 5 sonucu getir
    topk_df, scores = retrieve_top_k(query, df, embs)

    # 4) Sonuçları yazdır
    for rank, (idx, row) in enumerate(topk_df.iterrows(), 1):
        print(f"\n# {rank} - Sayfa {int(row['page'])}, skor {scores[rank-1]:.4f}\n{row['chunk']}")
