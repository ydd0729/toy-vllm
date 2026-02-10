# tiny-vllm
High performance LLM inference engine, a younger sibling of vLLM, written in C++ and CUDA

what I implement:

- load a LLM model from safetensors
- forward pass
- basic K/V cache
- PagedAttention
- batching

what I use:

- cuBLAS for all GEMMs
- tokenizer from hf transformers

some design choices:
- fp16
- goal is to maximize tok/s

### Is this AI-coded or human-coded

99% human-coded. It's a learning project for me, where learning >>> coding velocity. Some parts that are mundane can be AI-generated, like CMakeLists.txt or tokenization script

Jędrzej Maczan, 2026