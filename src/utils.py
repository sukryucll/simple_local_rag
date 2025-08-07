import textwrap
import os, sys

# 1 seviye yukarı çıkarak proje kökünü bul
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Şimdi config.py'yi doğrudan import edebilirsin:
from config import PDF_PATH, PAGES_CSV, CHUNKS_CSV, MAX_CHARS, OVERLAP


def print_wrapped(text, width=80):
    print(textwrap.fill(text, width))
