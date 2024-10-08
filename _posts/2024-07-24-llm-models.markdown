---
layout: sidebar-layout
title:  "LLM Models"
date:   2024-07-24 11:08:03 +0200
---

[Source Video](https://www.youtube.com/watch?v=bZQun8Y4L2A)

# Stages

* Pre-training -> Base model
* Supervised training -> SFT model (assistants, for example the ChatGPT is an assistent on top of the GPT4 base model)
* Reward modeling  -> RM model (humans choosing from multiple completions for a prompt, ranking them and then the model is * retrained to match the human's choice)
* Reinforcement learning -> RL model (based on the rewards given for each completion, lower the tokens part of a non rewarded completion and increase the rewarded ones)

[Source Video](https://www.youtube.com/watch?v=l8pRSuU81PU)

# GPT-2

A vocabulary of 50257 tokens.
Based on tiktoken.
Vercel app available to tokenize in the browser https://tiktokenizer.vercel.app/?model=gpt2

Based on UTF-8 encoding and byte-pair encoding algorithm:
* start with a string to tokenize
* get the utf-8 byte array for the string (elements between 0-255)
* count the occurences for each consecutive byte-pair
* for the pair with the max occurence, replace it with the next available ID (256 for the first replacement)
* now the vocabulary size is +1 but the input is shorter
* reiterate until you reach a desired vocabulary size

raw text <-> Unicode sequence <-> tokenizer <-> token sequence <-> LLM


# Embedding table

Encoding of the vocabulary: 50257 rows, one for each token, with ? columns, the dimension of the embedding vector.
Embeddings are trainable parameters of the model, similar tokens end up close to each other in the embeddings space.

# Attention layer

The context size is 1024 tokens so in the attention layer a token is connected to at most the previous 1023 tokens.

# Training speed ups

A Mac M2 has 3.6 TFLOPs (trillion floating point operations per second) on 160 execution units or 1280 ALUs.
Note: floating point is used instead of integers in order to model probability distributions.

Autocast not yet available in torch for MPS https://github.com/pytorch/pytorch/pull/99272 so can't use bflops16

[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/pdf/2307.08691)
[Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)

# DDP - Distributed Data Parallel

Distribute processing across multiple machines and GPUs.
Local rank: the rank of the GPU on a single machine, starting from 0.
Global rank: the rank of the GPU across all machines, starting from 0.

PyTorch provides DistributedDataParallel (DDP) which is a wrapper around torch.nn.parallel.DistributedDataParallel that handles the communication between processes.

A model is wrapped in DDP and then:
 - nothing changes on the forward pass
 - the backward pass is modified to handle gradients from multiple GPUs. It will aggregate (average) the gradients across all GPUs.

# Datasets

* Tiny Shakespeare
* WebText (Open WebText) - used  by OpenAI GPT-2 (~40GB); oudbound links from Reddit comments
* Common crawl - ~1TB of text (800M words)
* WebText2
* RedPajama, SlimPajama
* FineWeb, FineWebEdu
* Wikipedia

## Loading FineWebEdu dataset

[HiggingFace DataSet](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
Using Python dataset package:

1. load the dataset with dataset.load_dataset('HuggingFaceFW/fineweb-edu', split='train')
2. Tokenize each document in the dataset. Each tokenized document starts with <|endoftexttoken|> 
3. Load the tokens into numpy array of type uint64.
4. Output is saved to shards as numpy files, each with 100 million tokens.

When loading the tokens:
1. Load the numpy file.
2. Convert to torch.Tensor of type long.

# Validation

[HellaSwag](https://arxiv.org/abs/1905.07830)

A data set with sentences and 4 potential completions for each, where a single one is correct.

the sentence is feeded into the model and the probabilities for each of the 4 possible completions areevaluated. The highest probability should be for the correct answer. The individual token probabilities are averaged.
