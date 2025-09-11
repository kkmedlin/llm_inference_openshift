# benchmarks/benchmark.py

import time
import pandas as pd
from models.download_model import download_model
import torch
import os

def benchmark_model(model_name="distilgpt2", prompt="Hello world", max_new_tokens=50, run_id=None):
    # Load model and tokenizer
    model, tokenizer = download_model(model_name)
    device = torch.device("cpu")
    model.to(device)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Run inference and time it
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    end_time = time.time()

    latency = end_time - start_time

    # Decode output tokens to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save raw output to a text file (keeps full newlines)
    os.makedirs("benchmarks/outputs", exist_ok=True)
    txt_path = f"benchmarks/outputs/{model_name}_{run_id}_{prompt.replace(' ', '_')}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(generated_text)

    # Flatten text for CSV (so Excel doesn’t break rows)
    flat_text = generated_text.replace("\n", " ").replace("\r", " ")

    print(f"Inference latency: {latency:.3f} seconds")
    print(f"Generated text saved to {txt_path}\n")

    return {
        "model": model_name,
        "prompt": prompt,
        "latency_sec": latency,
        "output_text": flat_text
    }

if __name__ == "__main__":
    prompts = [
        "Hello world",
        "Once upon a time",
        "The future of AI is"
    ]

    results = []
    for i, p in enumerate(prompts, start=1):
        results.append(benchmark_model(prompt=p, run_id=i))

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Define output path
    out_path = "benchmarks/results.csv"

    # Append if file exists, else create new
    df.to_csv(
        out_path,
        mode="a",
        header=not os.path.exists(out_path),
        index=False
    )

    print(f"Benchmark results saved to {out_path}")
