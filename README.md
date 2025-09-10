# LLM Inference on OpenShift (CPU-Only)

*Work in Progress — this project is under active development.*

This repository explores **efficient deployment of lightweight large language models (LLMs)** in CPU-only environments using **OpenShift/Kubernetes**. The goal is to demonstrate practical approaches to **scalable inference** when GPU resources are unavailable.

## Motivation
Most LLM discussions assume access to powerful GPU clusters, but many real-world use cases require **CPU-only inference**.  
This repo investigates:
- Deploying quantized or distilled LLMs as microservices.  
- Benchmarking inference performance across runtimes (PyTorch, ONNX Runtime, `llama.cpp`).  
- Demonstrating horizontal scaling under simulated load on Kubernetes/OpenShift.  

## Planned Structure
```text
llm_inference_openshift/
│
├── README.md
├── deployment/
│   ├── Dockerfile            # container for quantized model inference
│   ├── k8s-deployment.yaml   # deployment config
│   ├── k8s-service.yaml      # service definition
│   └── helm-chart/           # optional, nice-to-have
├── models/
│   ├── download_model.py     # script to pull a small LLM (e.g. distilGPT2, quantized); caches model locally
│   └── run_inference.py      # simple API for CPU inference; runs demo
├── benchmarks/
│   ├── benchmark.py          # compare PyTorch vs ONNX Runtime vs llama.cpp
│   └── results.csv           # benchmark results
└── notebooks/
│   └── analysis.ipynb        # plots: latency, throughput, memory usage
├── reflections/
    └── learning.md           # daily reflection questions to promote learning


```

## Quickstart

For now, the repo contains a **basic inference demo** using Hugging Face Transformers on CPU.

1. Clone the repository:
   ```bash
   git clone https://github.com/kkmedlin/llm_inference_openshift.git
   cd llm_inference_openshift
   ```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run a test inference:
```bash
python models/run_inference.py
```

      
## Requirements
- Python 3.10+
- Tested on Ubuntu 22.04 and macOS (should work on WSL/Windows)

Core dependencies:
- [transformers](https://huggingface.co/transformers/)
- [torch](https://pytorch.org/)
- [onnxruntime](https://onnxruntime.ai/)
- numpy, pandas, matplotlib

See `requirements.txt` for full details.


## Roadmap
- [ ] Add ONNX Runtime inference support
- [ ] Build benchmarking script for CPU latency/memory usage
- [ ] Containerize with Docker + OpenShift deployment configs
- [ ] Add Helm chart for reproducible deployment
- [ ] Simulate load testing & autoscaling on OpenShift

## License
MIT License
