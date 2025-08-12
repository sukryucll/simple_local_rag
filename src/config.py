# src/config.py
import os

# Proje kök dizinini bul
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Dosya yolları
PDF_PATH       = os.path.join(DATA_DIR, "human-nutrition-text.pdf")
PAGES_CSV      = os.path.join(DATA_DIR, "pages.csv")
CHUNKS_CSV     = os.path.join(DATA_DIR, "chunks.csv")
EMBEDDINGS_CSV = os.path.join(DATA_DIR, "text_chunks_and_embeddings_df.csv")

# Chunk parametreleri
MAX_CHARS = 1200
OVERLAP   = 250

# Embedding modeli
HF_MODEL = "intfloat/e5-base-v2"
E5_USE_PREFIXES = False


# LLM yapılandırması
# Orijinal, hafif model: bloomz-560m
LLM_MODEL  = "bigscience/bloomz-560m"
MAX_TOKENS = 512

# Vector DB ayarları
CHROMA_DIR      = os.path.join(DATA_DIR, "chroma")   # ./data/chroma
CHROMA_COLLNAME = "nutrition"

# Reranker ayarları
USE_RERANKER = False   # True yaparsan main.py reranker kullanır
CANDIDATE_K  = 100     # Chroma'dan çekilecek aday sayısı
W_CE  = 1.0            # Cross-Encoder ağırlığı
W_SIM = 0.4            # Vektör benzerliği ağırlığı
W_LEX = 0.5            # Anahtar kelime eşleşmesi ağırlığı
