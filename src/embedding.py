# src/embedding.py
import os
import sys
import torch

# Proje kökünü bulup path'e ekle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import HF_MODEL, CHUNKS_CSV, EMBEDDINGS_CSV
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Manuel tokenizasyon + pooling ile embedding fonksiyonu
def embed_chunks_manual(
    chunk_csv: str = CHUNKS_CSV,
    out_csv: str   = EMBEDDINGS_CSV,
    model_name: str= HF_MODEL,
    pooling: str   = "mean"
) -> pd.DataFrame:
    """
    1. CHUNKS_CSV'i oku
    2. Tokenizer ve model yükle
    3. Tokenize et (padding, truncation)
    4. Modelden last_hidden_state al
    5. Mean pooling ile cümle embedding'i oluştur
    6. DataFrame'e ekle ve kaydet
    """
    # 1) Metin parçalarını oku
    df = pd.read_csv(chunk_csv)
    texts = df['chunk'].fillna(" ").astype(str).tolist()

    # 2) Tokenizer ve model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()

    # 3) Tokenize tüm metinleri
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # 4) Modelden çıkar
    with torch.no_grad():
        outputs = model(**tokens)
        hidden_states = outputs.last_hidden_state  # [B, L, D]

    # 5) Pooling
    mask = tokens['attention_mask'].unsqueeze(-1)  # [B, L, 1]
    masked_states = hidden_states * mask           # pad token'ları sıfırladı
    sum_states = masked_states.sum(dim=1)          # [B, D]
    lengths = mask.sum(dim=1)                     # [B, 1]
    embeddings = sum_states / lengths             # [B, D]

    # 6) DataFrame'e ekle
    emb_df = pd.DataFrame(embeddings.cpu().numpy())
    out_df = pd.concat([df[['page', 'chunk']].reset_index(drop=True), emb_df], axis=1)

    # 7) Kaydet
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"{len(texts)} chunks manually embedded → {out_csv}")
    return out_df

if __name__ == "__main__":
    embed_chunks_manual()
