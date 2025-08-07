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
MAX_CHARS = 3000
OVERLAP   = 200

# Embedding modeli
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM yapılandırması
# Orijinal, hafif model: bloomz-560m
LLM_MODEL  = "bigscience/bloomz-560m"
MAX_TOKENS = 512
