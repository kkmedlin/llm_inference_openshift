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
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Prompt:", prompt)
    print("Generated:", generated_text)

if __name__ == "__main__":
    main()
