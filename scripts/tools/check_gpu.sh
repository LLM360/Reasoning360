#!/usr/bin/env bash

# This script runs on a "single node" to check if there are "other users'" processes occupying the GPU

MY_UID=$(id -u)
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

# If there are no GPU processes, return 0 (indicating success)
if [ -z "$gpu_processes" ]; then
    echo "No GPU processes found on node $(hostname)."
    exit 0
fi

# If processes exist, check each one
echo "GPU processes found on node $(hostname):"
while IFS="" read -r pid; do
    pid=$(echo "$pid" | xargs)
    [ -z "$pid" ] && continue

    # Check process owner
    user_of_pid=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
    # Process might have exited during query
    if [ -z "$user_of_pid" ]; then
        continue
    fi

    # Get the user's UID
    uid_of_pid=$(id -u "$user_of_pid" 2>/dev/null)
    if [ -z "$uid_of_pid" ]; then
        continue
    fi

    # If UID doesn't match current script executor, it means someone else is using the GPU
    if [ "$uid_of_pid" != "$MY_UID" ]; then
        echo "  PID=$pid, user=$user_of_pid (UID=$uid_of_pid) is using GPU!"
        # Exit with non-zero status if someone else is using the GPU
        exit 1
    fi
done <<< "$gpu_processes"

# Exit normally if no other users were found after checking all processes
exit 0