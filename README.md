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

- [Intro: LLM, vLLM, models, inference servers, ???](#intro-llm-vllm-models-inference-servers-)
- [Technical prerequisities](#technical-prerequisities)
- [Safetensors and your model](#safetensors-and-your-model)
- [How floating-point numbers work and why we use bfloat16](#how-floating-point-numbers-work-and-why-we-use-bfloat16)
- [GPU and CPU memory](#gpu-and-cpu-memory)
- [Single token inference](#single-token-inference)
- [Prefill vs decode](#prefill-vs-decode)
- [GQA](#gqa)
- [Attention](#attention)
- [RoPE](#rope)
- [SiLU](#silu)
- [Residual connections](#residual-connections)
- [Causal mask](#causal-mask)
- [RMS Norm](#rms-norm)
- [Argmax](#argmax)
- [cublasGemmEx](#cublasgemmex)
- [The column-major to row-major transposition trick](#the-column-major-to-row-major-transposition-trick)
- [Why KV cache exists](#why-kv-cache-exists)
- [Buffer reuse](#buffer-reuse)
- [Static batching](#static-batching)
- [Continuous batching](#continuous-batching)
- [Online softmax](#online-softmax)
- [Paged Attention](#paged-attention)
- [Paged KV cache](#paged-kv-cache)
- [Paged Attention CUDA kernel](#paged-attention-cuda-kernel)


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

## How floating-point numbers work and why we use bfloat16

Let's remind ourselves about what we want to achieve. We want to load a model. We already know the structure of the file with a model, a Safetensors file. We know our reference model architecture. We checked that model weights are stored in BF16 type. Let's take a moment to think about this type and floating-point numbers in general

Everything on the computer is binary, at the very end. [Computer numbers formats, too](https://en.wikipedia.org/wiki/Computer_number_format).

In ML models, weights are almost never integer numbers. They are rather real numbers. And computers are binary. It means that people had to figure out how to represent real numbers in programming languages, in a way that is memory efficient. In this context, it means that you can pack a lot of information in small amount of bits. Because of course, you can build a complex data type ("complex" like in "non-trivial", not like in "real+imaginary"), where the part to the left from a dot separator is stored as an integer and the part after a separator (so the less than 1 part) is stored as another integer. But you can see how inefficient is that. It's only useful for situations when you really need full precision. And this is why things like [decimal in Python](https://docs.python.org/3/library/decimal.html) exist.

Ok, so now let's think what could we do instead. The simplest alternative would be to take an integer number, like 1234 and say "I will put a dot between 12 and 34", so you'd get 12.34. People solved this with a scaling factor, like literally, for 1234 you'd use scaling factor 1/100 to get 12.34. You still need to figure out how to represent it on the computer. In this format, the fractional part (after the dot) has always fixed length - it's exactly in the middle. This format is called [fixed-point numbers](https://en.wikipedia.org/wiki/Fixed-point_arithmetic). It's less widely used than floating-point numbers

So if there are fixed-point numbers, there inevitably need to be non-fixed-point numbers. Let's just think about it, loosely. Non-fixed-point could mean that the dot (the point) can move around. The fractional part can be longer or shorter. Sounds like more memory efficient solution.

Floating-point numbers, like all other numeric types are represented as a sequences of bits. All of them have slightly different design choices, so let's focus on [regular 16-bit float](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#IEEE_754_half-precision_binary_floating-point_format:_binary16) (float16, FP16, IEEE 754-2008).

Float16 and many other floating-point numbers work in a similar way. They consists of three sections: sign, exponent and fraction. In total they take 16 bits.

```
[ sign | exponent  | fraction            ]
[ 0    | 0 1 0 0 1 | 0 0 1 0 0 0 0 0 0 0 ]
```

The intuition is like this: in floating-point numbers you can move the dot around. Fraction is the number you use to move the dot around. Like in the previous example, you fraction can be 1234. This time, dot is not fixed, so it can be anywhere you choose - 1.234, 0.0001234, 1234000 etc.

Sign is a single bit. 0 means the number is positive and 1 means negative.

Exponent is 5 bits. It controls the size of the number.

Fraction is 10 bits. It controls what number we use to move the dot around. Alternative names: significand, mantissa.

Now the formula: $(-1)^{sign} * 2^{exponent-bias} * (1.fraction)$

Notice two new things - the $1.$ before the fraction and $bias$. The `1.` is a design choice to increase the number's precision (you get 1 bit for free) without storing it explicitly in the memory. That's why it's called implicit. See this [nice Stack Overflow question and answer](https://stackoverflow.com/questions/4930269/floating-point-the-leading-1-is-implicit-in-the-significand-huh) for a better explanation. The main idea is that it's not some must-have thing, but rather a clever trick to store more data without using more memory, through saying in the floating-points numbers specification that the bit is always there (except for zero). The next new thing is $bias$ and look again where it is: $2^{exponent-bias}$. Do you have any hunches about why is it there? 

The answer is that it's there to be able to represent smaller than 1 numbers too. And specifically, to do it by making it possible to have a negative power of $2$. Exponent is always a positive number, represented as a few bits (5 bits, in float16). If we didn't subtract bias from it, $2^{exponent}$ would be always positive, and perhaps quite a big number (binary 11111 is 31 decimal). So we could generate any number bigger than 1 or smaller than -1 (almost any number, limited by the precision of floating-point number), but we couldn't represent any small but bigger than 0 number, like 0.0001234. Hence, we need subtract bias from the exponent. Bias can't be neither too big nor too small. Knowing that, what do you think the bias for float16 is? Remember that max value of exponent is 31

So the bias for float16 is 15. It means you can represent both smaller and bigger numbers.

There are more interesting things in specification of floating-point numbers, like how do we encode positive and negative infinites, how do we encode not-a-number value etc. [Wiki has many examples about exponent encoding](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Exponent_encoding)

> The sizes of exponent and fraction are main differences between different types of floating-point numbers.

Let's take an example and encode number 12.34 as float16. 12.34 is positive, so sign bit is 0. To represent 12.34, we need to turn 12 into binary:

- 12 / 2 = 6, remainder 0
- 6 / 2 = 3, remainder 0
- 3 / 2 = 1, remainder 1
- 1 / 2 = 0, remainder 1

Read it backwards and you get 1100.

So decimal 12 is binary 1100. 

Now we turn the fraction 0.34 into binary:

- 0.34 * 2 = 0.68, integer part is 0
- 0.68 * 2 = 1.36, integer part is 1, we subtract 1 and continue
- 0.36 * 2 = 0.72, integer part is 0
- 0.72 * 2 = 1.44, integer part is 1, subtract 1 and continue
- 0.44 * 2 = 0.88, integer part is 0
- 0.88 * 2 = 1.76, integer part is 1, subtract 1 and continue
- 0.76 * 2 = 1.52, integer part is 1, subtract 1 and continue
- 0.52 * 2 = 1.04, integer part is 1, subtract 1 and continue
- 0.04 * 2 = 0.08, integer part is 0
- 0.08 * 2 = 0.16, integer part is 0

At this point, we have 10 bits computed, which is our length of fraction/mantissa/significand. At the first glance, it looks like we computed exactly as many bits as we need, but it's not entirely correct. It's because we need to include the first part of the number we encode - 12 (1100 in binary) - in the mantissa. $1.$ is always an implicit part of the number, so from $1100$ we remove the first $1$. We need to use what's left - $100$ - inside the mantissa. It means that we excessively computed 3 bits, which we will discard, because of 3 bits we need to use to include $12$ in the mantissa. So we didn't make a mistake, but we did some unnecessary work. Worth remembering if you decide to ever implement your own numeric type (and if you do it by yourself, [please let me know](jedrzej@maczan.pl)!)

Btw. look at the last term: "0.08 * 2 = 0.16, integer part is 0". The process is not finished, because we didn't reach 0 (we stopped at 0.16). It means that we will lose precision when representing this number.

Reading the bits from top to bottom, decimal 0.34 is binary approximately 0101011100.

Let's put 12 and 0.34 together and we get 1100 and 0101011100. Again, $1.0$ is implicit, so we're left with 100 and 0101011100. Put them together and you have 1000101011100. That's 13 bits. Remove 3 least significant bits (the ones to the right) and the final binary representation of fraction of our number is 1000101011.

> I know I'm sidetracking a bit, you were supposed to build an inference server. But honestly, why we need to rush? If these things are not interesting to you, then you can always skip forward. I believe there's some intristic joy of learning things in depth and if you're on the same page with me, let's continue

We're done with the mantissa. Look how did we turned $1100$ into $1.100$. If we were still in decimal world (base-10), we would say that to turn $1.100$ into $1100$ we need to multiply it by $1000$ ($10^3$). The numbers we operate on are binary (base-2), so it means that we need to multiply it by $2^3$, which immediately reminds us of $2^{exponent}$ from the floating-point formula. Which means that our exponent is $3$. But, the bias. Bias is 15. And in the formula for decoding the float, in the exponent of $2$ we subtract bias from the exponent that we read from floating-point bits (these 5 bits). Which means that we need to add the bias so it shows up in the stored exponent. So we add $3+15$ and our final exponent is $18$. Let's quickly turn 18 into binary format:

- 18 / 2 = 9, remainder 0
- 9 / 2 = 4, remainder 1
- 4 / 2 = 2, remainder 0
- 2 / 2 = 1, remainder 0
- 1 / 2 = 0, remainder 1

Read backwards and decimal 18 is 10010 binary.

We can put everything together to finally see decimal 12.34 as float16:

- sign bit: 0 (positive number)
- exponent: 10010 (18 binary, 3 because we shifted left by 3 positions + 15 bias; total 5 bits)
- fraction: 1000101011 (total 10 bits)

The number of bits correctly sums to 16. Together it looks like this:

```
[ sign | exponent  | fraction            ]
[ 0    | 1 0 0 1 0 | 1 0 0 0 1 0 1 0 1 1 ]
```

One interesting thing is that we know we lost some precision when turning 12.34 into a float. It was when we computed the 0.34 part. How about verify ourselves and decode our float back to decimal to see what value computer really stores when we try to store 12.34 as a float16? You curious, too?

Recall the formula: $(-1)^{sign} * 2^{exponent-bias} * (1.fraction)$

We know the number is positive, because sign is 0, so $-1^0 = 1$

We know the exponent is 18 minus bias 15 so it's 3. Plug it into the formula and we have: $2^3=8$

Fraction is 1000101011 and we need to turn it into decimal. We need to remember also about the implicit $1.$, so all together it's $1.1000101011$. Every next digit is multiplied by a power of 2, but with an exponent - 1 than the previous one. Similar to how it works when we turn a binary into decimal - we do the same but backwards, starting from the right we multiply each digit by 2 to the power of current position. Let's check it actually: exponent 10010 is then $0 * 1 + 1 * 2 + 0 * 4 + 0 * 8 + 1 * 16 = 18$, just like we thought. Ok, so now the mantissa. We turn $1000101011$ into decimal: $1 * 2^{-1} + 0 * 2^{-2} + 0 * 2^{-3} + 0 * 2^{-4} + 1 * 2^{-5} + 0 * 2^{-6} + 1 * 2^{-7} + 0 * 2^{-8} + 1 * 2^{-9} + 1 * 2^{-10} = 0.5 + 0 + 0 + 0 + 0.03125 + 0 + 0.0078125 + 0 + 0.001953125 + 0.0009765625 = 0.5419921875$. So putting it together with implicit $1.$, our number is $1.5419921875$.

Let's plug all of them into the formula together: $1 * 8 * 1.5419921875 = 12.3359375$. Correct and close enough, at least in LLM inference. The error: $12.34 - 12.3359375 = 0.0040625$

But this was float16. And our reference model (Llama 3.2 1B Instruct) used bfloat16 (BF16). It turns out bfloat16 is widely used in many different models. It also takes 16 bits, but it's exponent is longer (8 bits) than regular float16 (5 bits). First interesting thing about bfloat16 is that the size of exponent is actually the same as of twice bigger 32-bit float (float32, FP32, float), which also has 8 bit exponent. So comparing with float16, bfloat16 trades 3 bits of fraction to gain 3 bits of exponent. So naturally, the fraction shrinks and it's only 7 bits now. So to sum up - bfloat16 is 1 sign bit, 8 bit sign exponent and 7 bit fraction.

Why bfloat16 matters? It has the same size as float16 (a half size of a full precision float), and at the same time it has the same exponent as the float32 - at the cost of smaller fraction. The industry often chooses it for inference, because it's less likely to have range issues ([overflow](https://en.wikipedia.org/wiki/Integer_overflow) / [underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow)) and at the same time the loss of precision (due to smaller fraction) is an acceptable tradeoff in LLM inference (empirical results shows it).

If you want to test your understanding, Wikipedia page about half-precision floating-point formats has [lots of good examples](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Half_precision_examples)

Now you have all prerequisities to load a model in your inference engine and start doing interesing things. Would like to read some hands-on primer on working with GPU memory in CUDA? Keep reading the next section or skip to the inference part.

Oh btw, 32-bit floats are called single-precision floats. And then you hear about double-precision. And you might be like - omg so I get TWO floating points within the same number, then? Unfortunately the double precision just means that the number is bigger (64-bit). Meh

## GPU and CPU memory

This chapter might be a bit random, but I hope it's useful if you still begin with GPU programming. Data can live in host or device. 

Host is your PC with CPU. It has slow, big memory - DRAM - the one you buy and plugin into a motherboard. Recently very price'y, because of big tech AI craze. DRAM is separate from CPU. CPU has also it's own, small and fast on-chip memory - SRAM (I will wait until you stop laughing, if you're Slavic). 

Device is your GPU. It also has slow, big memory - HBM, VRAM - but it's soldered to the card and you can extend it easily, as you do with DRAM in your PC. It has small, fast memory - SRAM - which you can use if you defined `__shared__` variables.

GPU can't access your DRAM, so before running any computation on GPU, you need to copy data to it. The typical flow can be like this:

1. Create a variable on CPU
2. Write to variable on CPU
3. Compute the size of memory taken by the variable on CPU
4. Multiply the computed size by the size of variable type
5. Allocate the memory on GPU
6. Copy the variable from CPU to GPU
7. You can use this data in your GPU computations now

Ideally you want to allocate as little memory as possible, reuse this memory as much as possible and copy data as rarely as possible.

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

[Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) is one of the main operations used in deep learning, most notably in large language models. Matrix is a table of numbers. It has rows and columns. A single row and a single column is called a vector -- a sequence of numbers. Matrix multiplication uses two matrices, A and B, as an input and produces matrix C as an output. Matrix A has dimensions (M, K). Matrix B has dimensions (K, N). Both matrix A and B share the same dimension K. In other words, rows of matrix A have the same length as columns of matrix B. When you multiply A by B, you get a new matrix C with dimensions (M, N):

$$A (M,K) \times B(K,N)=C(M,N)$$

Every element of matrix C ($c_{ij}$) is a [dot product](https://en.wikipedia.org/wiki/Dot_product) of i-th row of A and j-th column of B. Dot product is a sum of all pairs, where each pair is a result of multiplying numbers from i-th row of A with numbers from j-th row of B on the same indices within their vectors (both row and column are vectors):

$$c_{ij} = a_{i0}b_{0j} + a_{i1}b_{1j}+...+a_{ik}b_{kj}=\sum_{x=0}^{k} a_{ix}b_{xj}$$

Back to large language models. Matrix multiplication happens when computing attention and Q, K and V projections. Most popular hardware to compute it efficiently are NVIDIA GPUs. They provide an important library, [cuBLAS](https://developer.nvidia.com/cublas), which allows you to run high performance linear algebra computations on their GPUs, including matrix multiplication, using [cublasGemmEx](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex) function.

## The column-major to row-major transposition trick

**TL;DR: if your data is in row-major format and you're going to use cuBLAS, then set transposition flag to `CUBLAS_OP_T` for matrices that are not tranposed yet, and `CUBLAS_OP_N` to matrices that are transposed in your formula**

Now the derivation and understanding:

The problem with cublasGemmEx is that it expects you to provide the matrices in [column-major format](https://en.wikipedia.org/wiki/Row-_and_column-major_order). And the LLMs, like Llama 3.2 1B Instruct, are distributed in row-major format.

![column and row major diagram](assets/column-row-major.png)

It turns out we don't have to modify the data format to use cuBLAS matrix multiplication functions. All thanks to these properties:

$$[A^T]_{ij}=[A]_{ji} \qquad C^T=B^T \times A^T \qquad (A^T)^T=A$$

The "$^T$" means that we transpose the matrix. Transposing a matrix turns columns into rows, and rows into columns. When you store the matrix in row-major format, and cuBLAS reads it in column-major format, it's an equivalent of transposing the matrix.

Let's see an example to understand it better: we want to compute $C = A \times B$, where A has dimensions (5, 2048) and B has dimensions (512, 2048). Our desired dimension of C is (5, 512). Right now, A and B dimensions are incompatible: $A(5, 2048)$ and $B(512, 2048)$. Do you remember that to get $C(M,N)$ we need $A(M,K)$ and $B(K,N)$? In other words, the second dimension of A and first dimension of B need to be equal. To achieve that, we need to transpose B. The formula becomes now: $C = A \times B^T$. The dimensions are ok now: $A(5,2048) \times B(2048, 512) = C(5, 512)$. Okay, so we would like to use cuBLAS now to compute the C.

But cuBLAS expects column-major format of A and B. Row-major transposed will give us column-major. So let's transpose the formula $C = A \times B^T$, using the property $C^T=B^T \times A^T$ and we get $C^T = (B^T)^T \times A^T$. From the third property of the matrix above, we know that a transposition of a transposition is equal to the original matrix, so we simplify $(B^T)^T$ to just $B$. The final formula is: $C^T = B \times A^T$. Let's check if dimensions are still correct. $B (512, 2048) \times A^T(2048, 5) = C^T(512, 5)$. The result dimensions seem inverse of what we wanted to get -- (5, 512) -- but notice that we still talk about $C^T$. The actual $C$, the output of the cublasGemmEx, is not transposed, so the final dimension is correct (5, 512). Now the code:

```cpp
cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, KV_DIM, num_active_slots, EMBEDDING_LENGTH, &k_proj_alpha, weights.w_k[layer], CUDA_R_16BF, EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH, &k_proj_beta, k_proj_batched_buffer, CUDA_R_16BF, KV_DIM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```

I want to preempt the last confusion you might have if you actually dig into the code. The flags `CUBLAS_OP_T` and `CUBLAS_OP_N` tell the cuBLAS which matrices to transpose. And we just derived the formula $C^T=B \times A^T$, so why do we now tell the cuBLAS to transpose the first matrix $B$? To understand it, think about column- / row-major again. From cuBLAS perspective, our row-major $B$ is transposed $B^T$, because cuBLAS reads it as if it were column-major. So we need to tell cuBLAS to transpose it, to get back the $B$ we derived. Similarly, since we derived that the second argument should be $A^T$, and cuBLAS reads row-major $A$ as a column-major $A^T$, then don't transpose it again, because it's how we wanted to provide it to the cublasGemmEx. Q.E.D. :D

> I will publish this section in slightly different form in [Paged Out! Issue #9 in the article "The cuBLAS transposition trick"](https://pagedout.institute/)

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