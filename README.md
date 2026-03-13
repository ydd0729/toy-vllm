# tiny-vllm
High performance minimal LLM inference engine, a younger sibling of vLLM, written in scratch C++ and CUDA

I build the project based on the vLLM paper [Efficient Memory Management for Large Language
Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)

- [x] load a real LLM model from safetensors (Llama 3.2 1B Instruct)
- [x] full LLM forward pass (prefill + decode)
- [x] all computation with CUDA kernels
- [x] KV cache
- [x] static batching
- [ ] continuous batching (in progress)
- [ ] PagedAttention

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

### model architecture
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
)
```

Jędrzej Maczan, 2026, Apache License 2.0