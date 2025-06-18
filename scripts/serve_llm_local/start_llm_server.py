#!/usr/bin/env python3
import argparse
import subprocess
import time
import re
import os
from typing import List, Dict
import json
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Launch multiple LLM server instances')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-32B', help='Name of the model to serve')
    parser.add_argument('--num_node_per_instance', type=int, default=1, help='Number of nodes per instance')
    parser.add_argument('--num_instances', type=int, default=4, help='Number of instances to start')
    parser.add_argument('--cancel', action='store_true', help='Cancel all running instances')
    return parser.parse_args()

def launch_server(model_name: str, num_node_per_instance: int) -> str:
    """Launch a single server instance and return its job ID"""
    cmd = ['sbatch', f'serve_single_model_{num_node_per_instance}node.sh', model_name]
    print(f"Launching server with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to launch server: {result.stderr}")
    
    # Extract job ID from output
    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from output: {result.stdout}")
    
    return match.group(1)

def save_job_ids(job_ids: List[str], server_info: List[Dict] = None, is_complete: bool = False):
    """Save job IDs to a JSON file"""
    timestamp = int(time.time())
    status = "complete" if is_complete else "incomplete"
    
    # Create server_status directory if it doesn't exist
    os.makedirs("server_status", exist_ok=True)
    
    output_file = f"server_status/server_instances_{status}_{timestamp}.json"
    
    # Create server info with full details if complete, otherwise just job IDs
    if is_complete and server_info:
        data_to_save = server_info
    else:
        data_to_save = [{"job_id": job_id} for job_id in job_ids]
    
    with open(output_file, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Saved {len(data_to_save)} server instances to {output_file}")
    return output_file

def get_server_ip(job_id: str) -> str:
    """Get the IP address of a running server instance"""
    # Wait for the server to start and get its IP
    max_retries = 30
    retry_interval = 10  # seconds
    
    for _ in range(max_retries):
        # Check if the job is still running
        cmd = ['squeue', '-j', job_id, '--noheader']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if not result.stdout.strip():
            raise RuntimeError(f"Job {job_id} is no longer running")
        
        # Try to get the IP from the log file
        log_file = f"slurm/serve_llm_as_verifier_{job_id}.log"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                # Look for the IP address in the log
                match = re.search(r'STEM_LLM_JUDGE_URL=http://([\d\.]+):8000', content)
                if match:
                    return match.group(1)
        
        time.sleep(retry_interval)
    
    raise RuntimeError(f"Could not get IP address for job {job_id} after {max_retries} retries")

def cancel_all_instances():
    """Cancel all running LLM server instances"""
    # Find all server instance JSON files
    json_files = glob.glob("server_status/server_instances_*.json")
    
    if not json_files:
        print("No server instance files found")
        return
    
    # Process all JSON files, not just the most recent one
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                server_info = json.load(f)
            
            print(f"\nProcessing file: {json_file}")
            print(f"Found {len(server_info)} server instances to cancel")
            
            for info in server_info:
                job_id = info['job_id']
                print(f"Cancelling job {job_id}...")
                subprocess.run(['scancel', job_id], check=True)
                print(f"Successfully cancelled job {job_id}")
            
            # Rename the JSON file to indicate it's been cancelled
            cancelled_file = json_file.replace('.json', '_cancelled.json')
            os.rename(json_file, cancelled_file)
            print(f"Server info moved to {cancelled_file}")
            
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

def main():
    args = parse_args()
    
    if args.cancel:
        cancel_all_instances()
        return
    
    # Launch all server instances
    job_ids = []
    for _ in range(args.num_instances):
        job_id = launch_server(args.model, args.num_node_per_instance)
        job_ids.append(job_id)
        print(f"Launched server instance with job ID: {job_id}")
    
    # Save job IDs immediately after submission
    save_job_ids(job_ids, is_complete=False)
    
    # Wait for all servers to start and collect their IPs
    server_info = []
    for job_id in job_ids:
        try:
            ip = get_server_ip(job_id)
            server_info.append({
                "job_id": job_id,
                "ip": ip,
                "url": f"http://{ip}:8000"
            })
            print(f"Server {job_id} is running at {ip}")
        except Exception as e:
            print(f"Error getting IP for job {job_id}: {e}")
    
    # Save complete server information
    save_job_ids(job_ids, server_info=server_info, is_complete=True)
    
    print("\nServer URLs:")
    for info in server_info:
        print(f"Job {info['job_id']}: {info['url']}")

if __name__ == "__main__":
    main()


# Example usage:
# python start_llm_server.py --model Qwen/Qwen3-32B --num_node_per_instance 1 --num_instances 4
# python start_llm_server.py --model deepseek-ai/DeepSeek-R1-0528 --num_node_per_instance 1 --num_instances 1
# python start_llm_server.py --model Qwen/Qwen3-235B-A22B --num_node_per_instance 1 --num_instances 1