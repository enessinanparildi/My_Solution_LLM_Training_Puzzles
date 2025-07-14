# 🔧 Multi-GPU LLM Training Puzzles: Pipeline + FSDP Simulation

This repository explores a minimalist and educational simulation of **Large Language Model (LLM)** training using **Pipeline Parallelism** and **Fully Sharded Data Parallel (FSDP)**. Designed as a learning tool, it breaks down complex training infrastructure into simple distributed async functions, making it ideal for studying model-parallel techniques in LLM training. The puzzles were originally developed by srush(https://github.com/srush).

---

## Features

- Simulated **Pipeline Parallelism** across model layers
- Lightweight **Fully Sharded Data Parallelism (FSDP)** across weight shards
- Gradient accumulation across multiple microbatches
- Simulated asynchronous communication (`pass_to`, `receive`, `allgather`)
- Layer and rank-aware compute distribution
- Optimizer update step after backpropagation
- Torch-free implementation — great for algorithm visualization
- Rank decomposition across shards and layers
- Model-parallel and data-parallel hybrid execution
- Manual forward/backward pass orchestration
- Weight sharding and merging with `allgather`
- Basic loss and gradient computation + update step
- Manual microbatch-based gradient accumulation
- Pipeline communication across pipeline stages
- Memory optimization via sharding and delayed allgather
- Simulation of FSDP-style allgather and shard logic
- Asynchronous scheduling and communication of activations and gradients


---

## Core Concepts

### `pipeline_fsdp(model: Model)`

The main async training loop simulates:

1. **Weight Shard Loading**:

   - Each rank loads its shard of weights using `load_weights`.

2. **Forward Pass**:

   - Activations are computed per layer and passed downstream using `pass_to`.

3. **Backward Pass**:

   - Gradients are computed and passed upstream.
   - Final layers compute loss; intermediate layers receive activations and pass back gradients.

4. **Gradient Accumulation**:

   - Gradients are accumulated across microbatches for each layer owned by the rank.

5. **Optimizer Update**:

   - Final accumulated gradients are used in `model.update()` to update weights.

---

## Techniques Implemented

- Layer and shard-based rank decomposition
- Forward pipeline execution using simulated async messaging
- Asynchronous activation passing between pipeline stages
- Simulated allgather to reconstruct full weights from shards (FSDP-style)
- Backward pipeline with upstream gradient communication
- Microbatch-based gradient accumulation and update
- Loss computation on final layer outputs
- Optimizer update using local gradient and state
- Layer-local activation and gradient storage with cleanup
- Shard-local optimizer state tracking and weight updates
- Fake gradients for unused layers to ensure completeness
- Hybrid parallelism combining pipeline and sharding
- Manual implementation of pipeline communication graph
- Modular layer-wise update hooks

---

## Educational Value

This repo is excellent for:

- Researchers prototyping custom parallelism strategies
- Engineers learning FSDP or pipeline parallel logic
- Educators explaining the internals of LLM training

---


