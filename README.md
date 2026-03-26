# tiny-vllm

You're going to build a high performance LLM inference engine with C++ and CUDA - tiny-vllm, a younger and smaller sibling of [vLLM]((https://arxiv.org/pdf/2309.06180))

We will learn a lot along the way, make mistakes and derive the ideas and maths from scratch

- [x] load a real LLM model from safetensors (Llama 3.2 1B Instruct)
- [x] full LLM forward pass (prefill + decode)
- [x] all computation with CUDA kernels
- [x] KV cache
- [x] static batching
- [x] continuous batching
- [x] [online softmax, FlashAttention-like](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [x] PagedAttention

Make yourself a hot beverage and let's begin


# The course below is in progress. ETA end of April 2026

## LLM, vLLM, models, servers, inference servers, ???

It's easy to get lost with so much going on recent years. Let's unpack it

**LLM is a model**. Physically, **LLM is a file which contains a lot of [float numbers](https://en.wikipedia.org/wiki/Floating-point_arithmetic)**. Conceptually, these numbers represent weights of operations. Weights are learned/discovered/found during training phase. Some of the operations use these weights. Every operation is a function, which takes some data as input, do something with it and produces data as output. Operations and their order are defined by LLM's architecture. Every model has its own architecture, which is designed by engineers and researchers.

The process of going from 0 to LLM writing a text is like this:

0. **Design the model** - engineers and researchers use high level language like Python with tensor library like [PyTorch](https://github.com/pytorch/pytorch) or [tinygrad](https://github.com/tinygrad/tinygrad) to design model's architecture. They train small versions of the model, make experiments with different operations, data and hyperparameters (parameters for operations). It's the phase of figuring out the specification
1. **Implement the model** - Once they decide on final model architecture and prepare the data for training, they write the code that defines the final model. It can be also in PyTorch or similar
2. **Train the model** - The chosen model architecture is initialized with dummy weights. They write a script which again uses PyTorch or similar to run learning algorithm like backpropagation on a lot of hardware, like [GPUs](https://en.wikipedia.org/wiki/Graphics_processing_unit) and [TPUs](https://en.wikipedia.org/wiki/Tensor_Processing_Unit). This phase burns a lot of energy, money and computational power. The product of training phase is a file with model weights, in some format, like [safetensors format](https://huggingface.co/docs/safetensors/index). So, the training phase is finding such a set of weights which produces good text using the given architecture
3. **Serve the model (we are here)** - The file with weights can't be ran on a computer. It's not an executable. It's a lot of numbers. The architecture can't be ran either - it's just a plan, a blueprint, a description of computation. To actually run the model, we need a program that turns the architecture and its operations into executable code and uses file with model weights to load the weights into the architecture. Once you write a program that implements the operations and once the program loads the weights (weights are loaded in the runtime of the program, at the startup), you can finally send prompts to the model and get meaningful response

> The training phase of an LLM is something we don't do in this course. We take a trained LLM and write a program which will run this LLM fast on NVIDIA GPU for multiple requests in parallel. If you want to train your own LLM, I strongly recommend sensei Karpathy repositories like [nanoGPT](https://github.com/karpathy/nanoGPT) and [llm.c](https://github.com/karpathy/llm.c) and his [YouTube channel](https://www.youtube.com/@AndrejKarpathy). Similarly, we don't design the model, but the tensor libraries are also fascinating topic and worth understanding from scratch. Geohot's [tinygrad](https://github.com/tinygrad/tinygrad) is a project which 

### model architecture
```py
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