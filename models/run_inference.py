# TODO:
# Load a small Hugging Face Model (e.g., distilgpt2)
# Run one sample inference (predict next word from a short prompt).
# Print output

# models/run_inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # Choose model (small, CPU-friendly)
    model_name = "distilgpt2"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure CPU-only
    device = torch.device("cpu")
    model.to(device)

    # Sample prompt
    promp

