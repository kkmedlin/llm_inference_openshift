# models/download_model.py
# by default, Hugging Face stores models in Windows: %USERPROFILE%\.cache\huggingface\transformers
# (e.g., C:\Users\kamed/.cache/huggingface/transformers)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import os

logging.set_verbosity_info()  # optional, prints info messages

def download_model(model_name="distilgpt2"):
    print(f"Downloading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Get Hugging Face cache path
    hf_cache = os.getenv("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers"))
    print(f"Model and tokenizer cached at: {hf_cache}")

    return model, tokenizer

if __name__ == "__main__":
    download_model()
