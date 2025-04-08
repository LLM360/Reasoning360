# stop all the slurm jobs by me with name containing "sgl-node"

#!/bin/bash

# Get all running jobs with name containing "sgl-node" submitted by the current user
echo "Searching for running 'sgl-node' jobs..."

# Get all running jobs with name containing "sgl-node" submitted by the current user
JOBS=$(squeue -u $USER | grep "sgl" | awk '{print $1}')

# Count the number of jobs
JOB_COUNT=$(echo "$JOBS" | wc -l)
echo "Found $JOB_COUNT running 'sgl-node' jobs."

# Ask for confirmation before cancelling
echo "Do you want to cancel all these jobs? (y/n)"
read -r CONFIRM

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    # Cancel each job
    for JOB_ID in $JOBS; do
        echo "Cancelling job $JOB_ID..."
        scancel "$JOB_ID"
    done
    echo "All jobs have been cancelled."
else
    echo "Operation cancelled. No jobs were stopped."
fi
