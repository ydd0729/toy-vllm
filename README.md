# tiny-vllm

You're going to build a high performance LLM inference engine with C++ and CUDA - tiny-vllm, a younger and smaller sibling of [vLLM](https://github.com/vllm-project/vllm)

We will learn a lot along the way, make mistakes and derive the ideas and maths from scratch

This repository consists of two things: 1. a full source code of the inference server and 2. a course where I lead you through the process of implementing the engine. Feel invited to use it as a learning tool on your learning path or if you are a lecturer, feel welcome to use it as a teaching resource at your university

The inference engine consists of:

- [x] load a real LLM model from Safetensors (Llama 3.2 1B Instruct)
- [x] full LLM forward pass (prefill + decode)
- [x] all computation with CUDA kernels
- [x] KV cache
- [x] static batching
- [x] continuous batching
- [x] [online softmax, FlashAttention-like](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [x] [PagedAttention](https://arxiv.org/pdf/2309.06180)

Make yourself a hot beverage and let's begin

# The course below is in progress

**ETA: end of April 2026**

After I finish text, I want to draw and add illustrations

## Intro: LLM, vLLM, models, inference servers, ???

It's easy to get lost with so much going on recent years. Let's unpack it

**LLM is a model**. Physically, **LLM is a file which contains a lot of [float numbers](https://en.wikipedia.org/wiki/Floating-point_arithmetic)**. Conceptually, these numbers represent weights of operations. Weights are learned/discovered/found during training phase. Some of the operations use these weights. Every operation is a function, which takes some data as input, do something with it and produces data as output. Operations and their order are defined by LLM's architecture. Every model has its own architecture, which is designed by engineers and researchers.

The process of going from 0 to LLM writing a text is like this:

0. **Design the model** - engineers and researchers use high level language like Python with tensor library like [PyTorch](https://github.com/pytorch/pytorch) or [tinygrad](https://github.com/tinygrad/tinygrad) to design model's architecture. They train small versions of the model, make experiments with different operations, data and hyperparameters (parameters for operations). It's the phase of figuring out the specification
1. **Implement the model** - Once they decide on final model architecture and prepare the data for training, they write the code that defines the final model. It can be also in PyTorch or similar
2. **Train the model** - The chosen model architecture is initialized with dummy weights. They write a script which again uses PyTorch or similar to run learning algorithm like backpropagation on a lot of hardware, like [GPUs](https://en.wikipedia.org/wiki/Graphics_processing_unit) and [TPUs](https://en.wikipedia.org/wiki/Tensor_Processing_Unit). This phase burns a lot of energy, money and computational power. The product of training phase is a file with model weights, in some format, like [Safetensors format](https://huggingface.co/docs/safetensors/index). So, the training phase is finding such a set of weights which produces good text using the given architecture
3. **Serve the model (we are here)** - The file with weights can't be ran on a computer. It's not an executable. It's a lot of numbers. The architecture can't be ran either - it's just a plan, a blueprint, a description of computation. To actually run the model, we need a program that turns the architecture and its operations into executable code and uses file with model weights to load the weights into the architecture. Once you write a program that implements the operations and once the program loads the weights (weights are loaded in the runtime of the program, at the startup), you can finally send prompts to the model and get a meaningful response. Generating an output from a model is called inference. That's why what we build here is called an inference server or inference engine

Knowing the reason behind a need for an inference server, let's think why we build it in C++ and CUDA. It's because we want to maximize efficient use of the hardware and get high performance. It means that we want to get responses fast and we want to be able to handle multiple prompts at the same time. CUDA is the whole ecosystem, but also a language that you use to write code that runs on GPUs. We need to write code on GPUs, because many operations inside LLM are multiplying and adding multiple numbers. If you need to do small amount of math, CPU enough. If a lot, GPU better. LLMs are mostly about multiplying the matrices, which boils down to computing dot products of two vectors, for many numbers and for many vectors. The math of LLMs is simple, we will need basics of linear algebra and you can learn while coding and fill the gaps on the go. I find this way of JIT learning the most effective and perhaps you will like it too

My take on a relationship between AI and computation which you maybe find useful is that **the intelligence comes from a lot of parameters of the model and a lot of computation of input values using these parameters**. There is no a single element, that you can point to and say: "this is what makes the model intelligent or useful". Every part of the model you can replace with a different one and get different tradeoffs in return, like trade accuracy for complexity.  I hope I won't forget to get back to this topic later, when we touch the math of attention. Because - the default attention mechanism is very computationally complex (O(n^2*d)). And this complexity can be challenged and in fact people do it and figure out alternative attention mechanisms, like [linear attention](https://haileyschoelkopf.github.io/blog/2024/linear-attn/). If more people will find this course useful, I will think about another one, about ML compilers (a practical one in Python or C++ + some SSA theory) or about alternative attention mechanisms (math + CUDA kernels). If you are interested, please let me know! If you will find this course valuable, please let other people know about it

> Out of scope: The training phase of an LLM is something we don't do in this course. We take a trained LLM and write a program which will run this LLM fast on NVIDIA GPU for multiple requests in parallel. If you want to train your own LLM, I strongly recommend sensei Karpathy repositories like [nanoGPT](https://github.com/karpathy/nanoGPT) and [llm.c](https://github.com/karpathy/llm.c) and his [YouTube channel](https://www.youtube.com/@AndrejKarpathy). Similarly, we don't design the model, but the tensor libraries are also fascinating topic and worth understanding from scratch. George Hotz's [tinygrad](https://github.com/tinygrad/tinygrad) is a project which implements a tensor library with a very little amount of code, so if you want to get inspired and learn the internals, it's a good place to do it (also [their Discord is nice](https://discord.com/invite/ZjZadyC7PK))! There is also a bit older and smaller version by Andrej Karpathy - [micrograd](https://github.com/karpathy/micrograd). And since I brought the Discord, I want to recommend you [Mark Saroufim's](https://www.marksaroufim.com/) [GPU MODE](https://discord.com/invite/gpumode). Many great people hanging out there! And if you feel lost with what is going on here, and you are new on your AI/ML journey, start with [Jeremy Howard and Rachel Thomas](https://www.fast.ai/about) [fastai book](https://course.fast.ai/Resources/book.html). I conveniently omit the data science and engineering part here, because I don't know much about it. Probably [Kaggle](https://www.kaggle.com/) can be a good place to start with it and learn on-hands. Last but not least, we're going to code in C++ and CUDA and use [cuBLAS](https://developer.nvidia.com/cublas) where applicable. You can learn on the go. NVIDIA [official resources](https://docs.nvidia.com/cuda/cuda-programming-guide/) are good and helpful 

## Technical prerequisities

You can build and run it on any platform, with minor changes, assuming you have a NVIDIA GPU. You might need to adjust some paths, like CUDA or GCC in [c_cpp_propertiesjson](.vscode/c_cpp_properties.json) or NVCC in [CMakeLists.txt](CMakeLists.txt)

I suggest you to fork this repo and make the necessary adjustments so it works on your machine and then create a pull request to [jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm) and upstream your changes for benefit of another readers

The exact setup on which I develop and test it:
- Linux (6.19.8 x64_64)
- [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (13.1)
- C++ 17
- [GCC](https://gcc.gnu.org/) (15.2.1)
- The only external dependency you will pull in is JSON parser [nlohmann/json](https://github.com/nlohmann/json) 3.12.0, which is a single header file [include/json.hpp](include/json.hpp)
- AMD CPU (Ryzen 7 9800X3D)
- NVIDIA GPU (RTX 5090)
- I used [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) from Hugging Face (commit hash 898999bd25b40516fce5a5b8f0948f4c81c650bc), you need just `model.safetensors` file from this repository

Install the dependencies and run the program with `./test.sh` - it will build it and immediately execute it

If you fail to build or run it and your AI of choice won't be able to help, please open an Issue on GitHub - I will try to help. Make sure to provide all useful context

## Safetensors and your model

First thing you need to do is to download a LLM which we will use to run inference on. I choose Llama 3.2 1B Instruct, because it's easy, small, tuned for dialogs and good enough for us. From perspective of us, the engineers who build an inference server, the model is just a single file containing weights.

The model is in [Safetensors format](https://huggingface.co/docs/safetensors/index). There exist other formats, like [Pickle](https://docs.python.org/3/library/pickle.html) and [Parquet](https://parquet.apache.org/docs/file-format/). Safetensors is just very popular and widely used, and the model we picked is hosted in Safetensors

Let's stop for a moment and understand the Safetensors format before we move on.

A safetensor file consists of 3 sections, always in this order: header size, header and tensors data. Header size is always 8 bytes. These 8 bytes are an unsigned 64-bit integer, which says how many bytes the actual header takes.

```cpp
std::ifstream safetensors_file("model.safetensors", std::ios_base::binary);
uint64_t header_size;
safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
```

The header is a JSON that contains about all the tensors inside the file. JSON is just a group of pairs <key, value>, where key is a unique string with a tensor name and value is another JSON object, containing info about this tensor. Every key in this JSON is a name of the tensor, except a single key which is called `__metadata__`, probably for some additonal info when necessary (we won't use it, specs say it's a "special key for storing free form text-to-text map). Every value is a JSON containing three keys - `dtype`, `shape` and `offsets`. `dtype` says what data type the tensor is stored in. `shape` says the [dimensions of a tensor](https://en.wikipedia.org/wiki/Tensor#As_multidimensional_arrays) and `offsets` say where the tensor is stored, within the tensors data section. Every `shape` is a list of ints of unknown length and every `offsets` value is a vector of exactly two ints. First element says where the tensor begins and last element says where the tensor ends.

You face the first design decision now. Do you want to make your inference server architecture independent, so it can run any arbitrary model, as long as you implement the operations it needs, or do you want to start simple and focus on our model of choice?

Whatever you decide, it's always easier to develop on a single model and then generalize, than try to make it flexible from the beginning when you're not sure how the code will look like at the end. You can always get back to it and update when you choose to.

If you want to make your server model-independent, you need to allocate the memory, set the model data `shape` and type (`dtype`) dynamically based on Safetensors header and implement more operations, making sure to cover all operations used by all models that you want to support. Probably you would still need to provide some blueprints of the models architecture, because the Safetensors file doesn't tell you which operation to run with this data, in what order etc. I'm not sure what's the optimal approach to do that, but if you figure it out, either on your own or by reading through vLLM/[TensorRT](https://github.com/NVIDIA/TensorRT)/others code, feel invited to share your findings

I will assume that we just code our server for Llama 3.2 1B Instruct architecture. Here's a dump of [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) `LlamaForCausalLM` object with `meta-llama/Llama-3.2-1B-Instruct` model loaded. We can inspect which operations we need to implement and what data shape and data types we need to use:

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

Let's dissect it.

First of all, we don't really know neither the order of operations or data type from this. But! [Model card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on Hugging Face page tell us that weights are in [BF16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) format. We will go back to the format soon. 

We need to understand the order of operations to know how to code it. Sebastian Raschka has a gallery of LLM architectures and it shows nicely how the operations are organized - see [here (the left one)](https://magazine.sebastianraschka.com/i/168650848/61-qwen3-dense).

By looking at the diagram from Sebastian, we see that the operations order in LLama 3.2 1B is like this:
1. Send some text to the model
2. Turn it into tokens (a new concept, we didn't mention it yet)
3. Retrieve an embedding for each token
4. 16 transformer blocks, also called layers, which consist of:
  - RMS Norm
  - Residual connection
  - Masked grouped-query attention, which consists of:
    - Q projection
    - K projection
    - V projection
    - RoPE with Q projection
    - RoPE with K projection
    - Attention
    - Attention scores
    - Causal mask
    - Softmax
    - Residual connection
    - Attention scores with V projection
  - O projection (output projection)
  - Residual connection add
  - RMS Norm
  - [Feed forward](https://en.wikipedia.org/wiki/Feedforward_neural_network) (like in first neural networks, [Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)), which consists of:
    - Gate, first linear layer
    - Up, second linear layer
    - [SiLU](https://arxiv.org/pdf/1702.03118) activation function, similar to [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) but looking more like a sigmoid
    - Down, third linear layer
    - Residual connection add
5. RMS Norm
6. Linear output
7. Argmax

After these steps, we should get our first token produced by the language model ran on our server.

Don't worry if some of this operations are not familiar yet. You will understand them viscerally once we progress through the course. I keep forgetting how they work, too, and often need to look up again, so don't feel bad for going back and forth or looking things up with a chatbot or internet search.

## How bfloat16 works

Let's remind ourselves about what we want to achieve. We want to load a model. We already know the structure of the file with a model, a Safetensors file. We know our reference model architecture. We checked that model weights are stored in BF16 type. Let's take a moment to think about this type and floating-point numbers in general

Everything on the computer is binary, at the very end. [Computer numbers formats, too](https://en.wikipedia.org/wiki/Computer_number_format).

In ML models, weights are almost never integer numbers. They are rather real numbers. And computers are binary. It means that people had to figure out how to represent real numbers in programming languages, in a way that is memory efficient. In this context, it means that you can pack a lot of information in small amount of bits. Because of course, you can build a complex data type ("complex" like in "non-trivial", not like in "real+imaginary"), where the part to the left from a dot separator is stored as an integer and the part after a separator (so the less than 1 part) is stored as another integer. But you can see how inefficient is that. It's only useful for situations when you really need full precision. And this is why things like [decimal in Python](https://docs.python.org/3/library/decimal.html) exist.

Ok, so now let's think what could we do instead. The simplest alternative would be to take an integer number, like 1234 and say "I will put a dot between 12 and 34", so you'd get 12.34. People solved this with a scaling factor, like literally, for 1234 you'd use scaling factor 1/100 to get 12.34. You still need to figure out how to represent it on the computer. In this format, the fractional part (after the dot) has always fixed length. This format is called [fixed-point numbers](https://en.wikipedia.org/wiki/Fixed-point_arithmetic). It's less widely used, but I didn't dig deep enough to tell you why

So if there are fixed-point numbers, there inevitably need to be non-fixed-point numbers. Let's just think about it, loosely. Non-fixed-point could mean that the dot (the point) can move around. The fractional part can be longer or shorter. Sounds like more memory efficient solution.

< In progress here >

If you want to test your understanding, Wikipedia page about half-precision floating-point formats has [lots of good examples](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Half_precision_examples)

Now you have all prerequisities to load a model in your inference engine and start doing interesing things. Would like to read some hands-on primer on working with GPU memory in CUDA? Keep reading the next section or skip to the inference part.

Oh btw, 32-bit floats are called single-precision floats. And then you hear about double-precision. And you might be like - omg so I get TWO floating points within the same number, then? Unfortunately the double precision just means that the number is bigger (64-bit). Meh

## Working with GPU memory

Incoming!

## Single token inference

Incoming!

## Prefill vs decode

Incoming!

## GQA

Incoming!

## Attention

Incoming!

## RoPE

Incoming!

## SiLU

Incoming!

## Residual connections

Incoming!

## Causal mask

Incoming!

## RMS Norm

Incoming!

## Argmax

Incoming!

## cublasGemmEx

Incoming!

## cuBLAS column-major trick

Incoming!

## Why KV cache exists

Incoming!

## Buffer reuse

Lifetimes analysis

Incoming!

## Static batching

Incoming!

## Continuous batching

Incoming!

## Online softmax

Math derivation

Incoming!

## Paged Attention

Incoming!

Memory management idea from operating systems used in LLM inference

## Paged KV cache

Incoming!

## Paged Attention CUDA kernel

Incoming!

Jędrzej Maczan, 2026, Apache License 2.0