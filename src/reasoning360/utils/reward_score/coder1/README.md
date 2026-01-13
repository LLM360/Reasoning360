# SandboxFusion Usage Guide

SandboxFusion is a secure code execution service that provides isolated environments for running untrusted code. This guide covers two deployment approaches: containerized deployment using SLURM (recommended for untrusted code) and local installation with lite isolation.

## 1. SLURM Container Deployment
*Recommended for executing untrusted or unknown code*

### Prerequisites
- Docker Hub access to pull the container image:
  ```bash
  enroot import docker://varad0309/code_sandbox:server
  # This creates: varad0309+code_sandbox+server.sqsh
  ```

### Deployment Steps

1. Create a SLURM batch script (`run_server.sbatch`):
```bash
#!/bin/bash
#SBATCH --job-name=sandbox_server
#SBATCH --output=logs/sandbox-%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=cpuonly
#SBATCH --nodelist=fs-mbz-cpu-XXX    # Replace XXX with your node
#SBATCH --time=4-00:00:00
#SBATCH --exclusive

srun --container-image=/path/to/varad0309+code_sandbox+server.sqsh  \
    --container-name=sandbox-server \
    --export=ALL,HOST=0.0.0.0,PORT=8080 \
    bash -c "make run-online"
```

2. Launch the service:
```bash
sbatch run_server.sbatch
```

3. Configure client:
   - Update the node address in `sandboxfusion_exec.py` to match your allocated SLURM node
   - This ensures clients can connect to your sandbox instance

## 2. Local Installation
*Suitable for trusted code with lite isolation*

### Prerequisites
- Conda environment manager
- Poetry package manager

### Installation Steps

1. Get the code:
```bash
git clone https://github.com/bytedance/SandboxFusion.git 
cd SandboxFusion
```

2. Configure and install:
```bash
# Important configuration changes:
# 1. In sandbox/configs/local.yaml:
#    - Set isolation: "lite"
#    - Set max_concurrency: 0 (uses all available resources)

poetry install
mkdir -p docs/build

# Optional: Install Python runtime
cd runtime/python
# Note: Edit install-python-runtime.sh to skip requirements.txt if needed
bash install-python-runtime.sh
```

3. Start the server:
```bash
make run-online
```

## Using the Service

The API interface is identical for both deployment methods:

```python
from verl.utils.reward_score.coder1.sandboxfusion_exec import code_exec_sandboxfusion

# Basic code execution
code = 'print("Hello, World!")'
success, output = code_exec_sandboxfusion(code, stdin="", timeout=5)

# Test-driven execution
code = 'def add(a, b): return a + b'
test_code = '''
def test_add():
    assert add(2, 3) == 5
'''
success, output = code_exec_sandboxfusion_with_pytest(code, test_code, timeout=5)
```

## Scaling and Load Balancing

For high-throughput requirements:
- Deploy multiple instances across different nodes
- The service automatically handles load distribution
- Each instance can process multiple concurrent requests

## Security Features

1. Execution Isolation:
   - SLURM container: Full isolation with resource limits
   - Local mode: Lite isolation for trusted code

2. Resource Controls:
   - Memory limits
   - CPU allocation
   - Network access restrictions
   - Filesystem isolation

## Performance Benchmarks

We evaluated different execution methods on two datasets:
- Leetcode 2k: A collection of 2,000 coding problems
- Taco 13k: A larger dataset with 13,000 problems

| Method | Leetcode 2k | Taco 13k |
|--------|------------|-----------|
| Subprocess (baseline) | 25 sec | 1369 sec |
| SandboxFusion | 26 sec | 1353 sec |
| SandboxFusion + enroot | 70 sec | 6385 sec |

Key observations:
1. SandboxFusion with isolation maintains performance comparable to baseline subprocess
2. Adding container isolation (enroot) increases execution time but provides maximum security
3. Choose deployment method based on your security vs performance requirements:
   - High security: Use containerized deployment
   - Balanced: Use SandboxFusion with isolation
   - Performance critical: Consider subprocess (only for trusted code)

## Troubleshooting Guide

### Common Issues and Solutions

1. Event Loop Errors
   - Symptom: FastAPI TestClient errors
   - Solution: Switch to direct requests library
   - Note: This typically occurs in multi-threaded scenarios

2. Container Access Issues
   - Check SLURM node allocation
   - Verify container paths and permissions
   - Ensure network connectivity to the node

3. Performance Optimization
   - Adjust SLURM resource allocation (CPU, memory)
   - Monitor system resource usage
   - Consider node-local storage for I/O-intensive operations
