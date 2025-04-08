#!/bin/bash

# Set environment variables
export PYTHONUNBUFFERED=1

# Define dataset name
DATASET=big-math-rl-verified_problems

# Create output directory if it doesn't exist
WORKING_DIR=${HOME}/Reasoning360
OUTPUT_DIR=${WORKING_DIR}/sgl_inference_results/${DATASET}
TEMP_JOB_DIR=${WORKING_DIR}/temp_job_scripts
mkdir -p ${OUTPUT_DIR}
mkdir -p slurm
mkdir -p ${TEMP_JOB_DIR}

# Create a single-node job script template
cat > ${TEMP_JOB_DIR}/single_node_job.sh << EOF
#!/bin/bash
#SBATCH --job-name=sgl-node-%NODE_ID%
#SBATCH --partition=mbzuai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=slurm/sgl-%NODE_ID%-%j.out
#SBATCH --error=slurm/sgl-%NODE_ID%-%j.err
#SBATCH --time=48:00:00

# Activate conda environment
source /mbz/users/shibo.hao/miniforge3/etc/profile.d/conda.sh
conda activate sglang

# Define paths and variables
WORKING_DIR=\${HOME}/Reasoning360
DATASET="${DATASET}"
DATA_DIR=\${WORKING_DIR}/\${DATASET}
OUTPUT_DIR=\${WORKING_DIR}/sgl_inference_results/\${DATASET}
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BASE_PORT=30000

# Get node ID from environment variable
NODE_ID=%NODE_ID%
HOSTNAME=\$(hostname)
PORT=\$((BASE_PORT + NODE_ID))

echo "Running job for node \${NODE_ID} on \${HOSTNAME}, using port \${PORT}"
echo "Using dataset: \${DATASET}"

# Create node-specific output directory
NODE_OUTPUT_DIR="\${OUTPUT_DIR}/node_\${NODE_ID}"
mkdir -p \${NODE_OUTPUT_DIR}

# Set a unique cache directory for each job to avoid SQLite locking issues
export OUTLINES_CACHE_DIR=/tmp/.outlines_node_\${NODE_ID}

# Start SGLang server on this node
echo "Starting SGLang server with model: \${MODEL_PATH}"

python -m sglang.launch_server \\
    --model-path \${MODEL_PATH} \\
    --host 0.0.0.0 \\
    --port \${PORT} \\
    --max-running-requests 512 \\
    --data-parallel-size 8 &

# Store the server process ID
SERVER_PID=\$!

# Wait for server to start
echo "Waiting for server to initialize (300 seconds)..."
sleep 300

# Get the actual IP address of this node
NODE_IP=\$(hostname -I | awk '{print \$1}')
echo "Node IP address: \${NODE_IP}"

# Run batch inference on this node
echo "Starting batch inference on split_\${NODE_ID}.jsonl"
python \${WORKING_DIR}/data_analysis/batch_inference.py \\
    --input-file \${DATA_DIR}/split_\${NODE_ID}.jsonl \\
    --output-folder \${NODE_OUTPUT_DIR} \\
    --chunk-size 32768 \\
    --endpoint "/v1/chat/completions" \\
    --completion-window "24h" \\
    --server-url "http://\${NODE_IP}:\${PORT}/v1"

# Check if batch inference completed successfully
if [ \$? -eq 0 ]; then
    echo "Batch inference completed successfully on node \${NODE_ID}"
else
    echo "Batch inference failed with error code \$? on node \${NODE_ID}"
fi

# Terminate the server
echo "Shutting down SGLang server on node \${NODE_ID}"
kill \${SERVER_PID}

echo "Job completed for node \${NODE_ID}"
EOF

# Launch 64 separate sbatch jobs
JOB_IDS=()
for NODE_ID in {0..63}; do
    # Create a node-specific job script by replacing the placeholder
    sed "s/%NODE_ID%/${NODE_ID}/g" ${TEMP_JOB_DIR}/single_node_job.sh > ${TEMP_JOB_DIR}/node_${NODE_ID}_job.sh
    chmod +x ${TEMP_JOB_DIR}/node_${NODE_ID}_job.sh
    
    # Submit the job and capture the job ID
    echo "Submitting job for node ${NODE_ID}..."
    JOB_ID=$(sbatch --parsable ${TEMP_JOB_DIR}/node_${NODE_ID}_job.sh)
    JOB_IDS+=($JOB_ID)
    echo "Submitted job ID: $JOB_ID for node ${NODE_ID}"
    
    # Optional: add a small delay between submissions to avoid overwhelming the scheduler
    sleep 1
done

# Save job IDs to a file for later reference if needed
echo "${JOB_IDS[@]}" > ${WORKING_DIR}/sgl_job_ids.txt
echo "All job IDs saved to ${WORKING_DIR}/sgl_job_ids.txt"

# Clean up the template file
rm ${TEMP_JOB_DIR}/single_node_job.sh

echo "All 64 jobs have been submitted."
echo "Job scripts are stored in ${TEMP_JOB_DIR} and can be removed after all jobs complete."

# Start monitoring immediately
echo "Starting job monitoring..."
echo "Press Ctrl+C to stop monitoring at any time."
echo "Job IDs are stored in ${WORKING_DIR}/sgl_job_ids.txt and can be modified manually."
echo ""
sleep 2  # Brief pause to let the user read the message

# Directly implement monitoring logic
echo "Monitoring jobs from ${WORKING_DIR}/sgl_job_ids.txt..."
while true; do
    clear
    echo "=== SGLang Job Monitoring $(date) ==="
    echo "Dataset: ${DATASET}"
    echo ""
    
    # Check job status
    RUNNING=0
    COMPLETED=0
    FAILED=0
    PENDING=0
    
    # Read job IDs from file, handling space-separated format
    for JOB_ID in $(cat "${WORKING_DIR}/sgl_job_ids.txt"); do
        # Skip empty entries
        [[ -z "$JOB_ID" ]] && continue
        
        STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
        NODE_ID=$(squeue -j $JOB_ID -h -o "%j" 2>/dev/null | grep -o '[0-9]\+' || echo "unknown")
        
        if [ -z "$STATUS" ]; then
            # Job not in queue, check if it completed successfully
            if sacct -j $JOB_ID -n -P -X --format=state | grep -q "COMPLETED"; then
                STATUS="COMPLETED"
                COMPLETED=$((COMPLETED+1))
            else
                STATUS="FAILED/CANCELLED"
                FAILED=$((FAILED+1))
            fi
        elif [ "$STATUS" == "RUNNING" ]; then
            RUNNING=$((RUNNING+1))
        elif [ "$STATUS" == "PENDING" ]; then
            PENDING=$((PENDING+1))
        fi
        
        # Check progress by counting output files
        NODE_OUTPUT_DIR="${OUTPUT_DIR}/node_${NODE_ID}"
        if [ -d "$NODE_OUTPUT_DIR" ]; then
            FILE_COUNT=$(find "$NODE_OUTPUT_DIR" -type f -name "*.jsonl" | wc -l)
            echo "Job $JOB_ID (Node $NODE_ID): $STATUS - Files: $FILE_COUNT"
        else
            echo "Job $JOB_ID (Node $NODE_ID): $STATUS - No output directory yet"
        fi
    done
    
    echo ""
    echo "Summary: $RUNNING running, $PENDING pending, $COMPLETED completed, $FAILED failed"
    echo ""
    echo "Press Ctrl+C to exit monitoring"
    echo "You can modify ${WORKING_DIR}/sgl_job_ids.txt to change which jobs are monitored"
    
    # Exit if all jobs are done
    if [ $RUNNING -eq 0 ] && [ $PENDING -eq 0 ]; then
        echo "All jobs have completed or failed. Exiting monitor."
        break
    fi
    
    sleep 10
done