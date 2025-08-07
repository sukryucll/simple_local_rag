# src/prompting.py
import os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# Proje kökünü path'e ekle
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Ayarları yükle
from config import LLM_MODEL, MAX_TOKENS

# Model ve tokenizer yüklemesi
print(f"Loading LLM model: {LLM_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model     = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

def format_prompt(query: str, contexts: list[str]) -> str:
    """
    Create an English prompt using the provided context list.
    """
    header = "Answer the question using the following context:\n\n"
    ctx    = "\n\n".join(contexts)
    return f"{header}{ctx}\n\nQuestion: {query}\nAnswer:"


def ask_llm(prompt: str, max_new_tokens: int = MAX_TOKENS) -> str:
    """
    Sends the prompt to the model and returns just the generated answer,
    stripping off the prompt if it's repeated.
    """
    inputs  = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    full    = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if full.startswith(prompt):
        return full[len(prompt):].strip()
    return full.strip()
