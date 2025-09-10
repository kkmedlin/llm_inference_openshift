# Daily Reflection Questions

This file contains questions I ask myself each day while working on this project. 
I’m using ChatGPT as a learning and guidance tool to reflect on design decisions, implementation, and ML principles.

## Day 1 Questions

### Project Design
- Why did I choose this model (e.g., distilgpt2) for CPU-only inference? distilgpt2 is a smaller (architecture), faster version of GPT-2. It was trained using a method called "knowledge" distillation in which GPT2 is the "teacher" and distilgpt2 is the "student." Developed by Hugging Face, distilgpt2 is easily accessible and compatible with Hugging Face Transformers library and its ecosystem. PROS: generative capabilities. strong balance between speed and acuraccy. good for games. CONS: potential for bias. not fact-sensitive. limited context understanding.

- How could I improve model efficiency without using a GPU?
(a) Distill the model: as decribed above for what ditilgpt2 is to GPT2, and/or (b) Quantize the model: Reduce the level floating-point number precision; e.g. from 32-bit to 8-bit integers. NOTE: Distilation reduces memory footprint while quanitzation reduces storage size.
- What are the trade-offs of quantization vs distilled models?

Distilled models can be less fact-sensitive, more biased, and use less context for understanding. Quantized models can lead to a reduction in model accuracy due to the lowering of precision. NOTE: bias and context limiations are important if later you evaluate performance on sensitive or real-world prompts.

### Implementation
- Did I follow best practices for Python, modularity, and code readability?
I believe so. The scripts are well-documents and the files well organized.

- Are there any parts of the code I could refactor to be cleaner or more reusable?
The only script there is is run_inference.py and download_model.py. They are short and appear well-documented to me.

- How does caching the model locally affect workflow and performance?
Essentially, it downloads model parameters and an instance of the model onto your local machine. This way you can run subsequent iterations locally, which streamlines workflow and improves performance. One less road to travel. NOTE: Caching helps offline work, and speeds up iterative development.

### Benchmarks / Performance
- What patterns do I notice in latency measurements?
I still need to do this...
- Which runtime (PyTorch, ONNX) seems more efficient for CPU?
OOO, this seems like a fund question. Let's create a test.
- How could I structure benchmarks to be more informative?
Got to seem them first.

### Learning & Reflection
- What new concepts did I learn today about LLM inference or OpenShift?
- How did using ChatGPT help me understand or make decisions today?
- What would I do differently if I started this step over tomorrow?
