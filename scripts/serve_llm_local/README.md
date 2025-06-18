# LLM Server Management Scripts

Two Python scripts for launching and using local LLM server instances on a SLURM cluster.

## Overview

- **`start_llm_server.py`**: Orchestrates multiple LLM server instances on a cluster environment
- **`call_local_llm.py`**: Asynchronous client for processing datasets through the local LLM servers

## Quick Start

### 1. Launch LLM Servers

```bash
# Launch 4 instances of Qwen3-32B model with 1 node each
python start_llm_server.py --model Qwen/Qwen3-32B --num_node_per_instance 1 --num_instances 4

# Launch 4 instances of DeepSeek-R1 model with 2 nodes each
python start_llm_server.py --model deepseek-ai/DeepSeek-R1-0528 --num_node_per_instance 2 --num_instances 4

# Launch single instance of large model
python start_llm_server.py --model Qwen/Qwen3-235B-A22B --num_node_per_instance 1 --num_instances 1
```

### 2. Process Dataset

Wait for servers to be ready, then run:

```bash
python call_local_llm.py
```

### 3. Cancel All Servers

```bash
python start_llm_server.py --cancel
```

## Configuration

### Server Launcher Arguments

- `--model`: Name of the model to serve (default: `Qwen/Qwen3-32B`)
- `--num_node_per_instance`: Number of nodes per instance (default: 1)
- `--num_instances`: Number of instances to start (default: 4)
- `--cancel`: Cancel all running instances

### Client Configuration

Edit these parameters at the top of `call_local_llm.py`:

```python
TIMEOUT_SECONDS = 24000           # 400 minutes total timeout
MAX_WINDOW_SIZE_PER_INSTANCE = 256  # Max concurrent requests per server
NUM_GENERATIONS = 16              # Number of generations per input
MODEL = "deepseek-ai/DeepSeek-R1-0528"  # Model identifier
```

### Input Data

The client reads from `~/libra-eval/libra_eval/datasets/iq_250.jsonl` by default. Format:

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}]}
{"messages": [{"role": "user", "content": "Explain quantum computing"}]}
```

To use a different dataset, modify this line in `call_local_llm.py`:
```python
data_df = pd.read_json("path/to/your/dataset.jsonl", lines=True)
```

## Output

- **Server status**: Saved in `server_status/` directory as JSON files
- **LLM responses**: Saved to `llm_responses.jsonl` with original data plus responses

## Prerequisites

- SLURM batch system with appropriate batch script templates
- Python 3.7+ with `pandas` and custom `utils` module
- Running LLM server instances for the client script 