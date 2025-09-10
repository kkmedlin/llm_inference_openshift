# benchmarks/benchmark.py
import time
import pandas as pd
from models.download_model import download_model
import torch

def benchmark_model(model_name="distilgpt2", prompt="Hello world", max_new_tokens=50):
    # Load model and tokenizer
    model, tokenizer = download_model(model_name)
    device = torch.device("cpu")
    model.to(device)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Run inference and time it
    start_time = time.time()
    _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
    end_time = time.time()

    latency = end_time - start_time
    print(f"Inference latency: {latency:.3f} seconds")

    return {"model": model_name, "prompt": prompt, "latency_sec": latency}

if __name__ == "__main__":
    # Example prompts
    prompts = [
        "Hello world",
        "Once upon a time",
        "The future of AI is"
    ]

    results = []
    for p in prompts:
        results.append(benchmark_model(prompt=p))

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("benchmarks/results.csv", index=False)
    print("Benchmark results saved to benchmarks/results.csv")

