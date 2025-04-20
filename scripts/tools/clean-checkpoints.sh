# A script to clean the checkpoints that are not multiples of 24
# (in case the saving frequency was set too high and the disk space is limited)

base_dir="checkpoints/Reasoning360/shibo-math-grpo-32nodes-setting3-Qwen2.5-32B"
for folder in "$base_dir"/global_step_*; do
    # Skip if no files match the pattern
    [ -e "$folder" ] || continue
    
    # Extract the step number from the filename
    echo "Folder: $folder"
    step_number=$(basename "$folder" | grep -o 'global_step_[0-9]*' | grep -o '[0-9]*')
    echo "Step number: $step_number"
    
    # Check if step_number is not a multiple of 24
    if [ ! -z "$step_number" ] && [ $((step_number % 24)) -ne 0 ]; then
        rm -rf "$folder"
        echo "Removed: $folder"
    fi
done