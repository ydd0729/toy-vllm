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
- Test on Llama 3.2 1B
- BF16 (because Llama 3.2 1B uses it)
- Single GPU (tested on RTX 5090 32GB)

### use

```
python python/tokenizer.py "The capital of France is" | ./tiny-vllm
```

Jędrzej Maczan, 2026, Apache License 2.0