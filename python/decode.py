from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# tokens = [
#     12366,
#     315,
#     6680,
#     315,
#     14924,
#     315,
#     16309,
#     315,
#     7908,
#     791,
#     7908,
#     791,
#     791,
#     791,
#     791,
#     791,
#     791,
#     791,
# ]
tokens = [
    271, 49974, 388, 791, 14924
]
print(t.decode(tokens))
