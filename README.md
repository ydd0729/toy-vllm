# tiny-vllm
High performance minimal LLM inference engine, a younger sibling of vLLM, written in scratch C++ and CUDA

I build the project based on the vLLM paper [Efficient Memory Management for Large Language
Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)

- load a LLM model from safetensors
- full LLM forward pass
- CUDA kernels for attention etc
- KV cache as described in the paper
- PagedAttention
- batching

External libraries:

- cuBLAS for all GEMMs
- tokenizer from HuggingFace Transformers

Main design choices:
- FP16
- Test on Llama 3.2 1B
- Single GPU (tested on RTX 5090 32GB)

Jędrzej Maczan, 2026, Apache License 2.0