# main.py (root)
import os
import sys
import pandas as pd

# 1) src klasörünü path'e ekle
ROOT    = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 2) Ayarları yükle
from config import PAGES_CSV, CHUNKS_CSV, EMBEDDINGS_CSV
from ingestion import load_pdf_pages
from chunking import chunk_pages
from embedding import embed_chunks_manual
from retrieval import retrieve_top_k
from prompting import format_prompt, ask_llm


def prepare_data():
    # PDF → pages.csv
    if not os.path.exists(PAGES_CSV):
        print("→ pages.csv bulunamadı, PDF'ten çıkarılıyor...")
        pages = load_pdf_pages()
        os.makedirs(os.path.dirname(PAGES_CSV), exist_ok=True)
        pd.DataFrame({"text": pages}).to_csv(PAGES_CSV, index=False, encoding="utf-8")
    else:
        pages = pd.read_csv(PAGES_CSV)["text"].tolist()

    # pages.csv → chunks.csv
    if not os.path.exists(CHUNKS_CSV):
        print("→ chunks.csv bulunamadı, parçalar oluşturuluyor...")
        chunks = chunk_pages(pages)
        os.makedirs(os.path.dirname(CHUNKS_CSV), exist_ok=True)
        pd.DataFrame(chunks).to_csv(CHUNKS_CSV, index=False, encoding="utf-8")
    else:
        df_chunks = pd.read_csv(CHUNKS_CSV)
        chunks = df_chunks.to_dict(orient="records")

    # chunks.csv → embeddings.csv
    if not os.path.exists(EMBEDDINGS_CSV):
        print("→ embeddings.csv bulunamadı, embedding yapılıyor...")
        emb_df = embed_chunks_manual()
    else:
        emb_df = pd.read_csv(EMBEDDINGS_CSV)

    return emb_df


def run(query: str, top_k: int = 5):
    # Verileri hazırla veya yükle
    emb_df = prepare_data()
    
    # Retrieval için embedding matrisini hazırla
    embs = emb_df.drop(columns=["page", "chunk"]).values

    # En iyi top_k chunk seç
    topk_df, scores = retrieve_top_k(query, emb_df, embs, k=top_k)

    # Retrieved contexts
    print("\n=== Retrieved Contexts ===")
    for i, (idx, row) in enumerate(topk_df.iterrows()):
        score = scores[i]
        print(f"\n#{i+1} - Page {int(row['page'])}, score {score:.4f}\n{row['chunk']}")

    # Prompt & LLM cevap
    contexts = topk_df["chunk"].tolist()
    prompt   = format_prompt(query, contexts)
    answer   = ask_llm(prompt)

    # Cevabı göster
    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    q = input("Your question: ")
    run(q)
