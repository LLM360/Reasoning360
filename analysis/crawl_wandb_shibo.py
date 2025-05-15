import wandb
import pandas as pd
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def crawl_wandb_data(project_name, entity, metric_names, run_ids=None):
    """
    Crawl wandb data for specific runs and metrics.
    
    Args:
        project_name (str): Name of the wandb project
        metric_names (str or list): Name(s) of the metric(s) to extract
        entity (str, optional): The entity (username or team name). Defaults to None.
        run_name_filter (str, optional): Filter to apply to run names. Defaults to None.
        run_name_list (list, optional): List of run names to crawl. Defaults to None.
    
    Returns:
        dict: Dictionary of run_name -> DataFrame with metric data
    """
    # Initialize wandb API
    print("run_ids: ", run_ids)
    api = wandb.Api()
    
    # Convert single metric to list for consistent handling
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    # Get all runs from the project that are in running state
    # Note: We're not filtering by run_name_filter to keep all running experiments
    project_path = f"{entity}/{project_name}" if entity else project_name
    run_data = {}
    
    # keep only running or finished runs
    # filters = {"state": {"$in": ["running", "finished"]}}
    runs = api.runs(project_path)
    
    print(f"Found {len(runs)} runs")
    
    for run in tqdm(runs):
        
        if run.id not in run_ids:
            continue
        
        try:
            print(f"Processing run {run.name}")

            # Get a sample history to identify available keys
            # Fetching just one step is enough to get column names
            sample_history = run.history(samples=2)
            print(f"Sample history: {sample_history}")

            # Identify metrics that start with "val/"
            print(f"Sample history columns: {sample_history.columns}")
            save_metrics = [key for key in sample_history.columns if any([key.startswith(m) for m in metric_names])]
            print(f"Found metrics in run {run.name}: {save_metrics}")

            # Get the full history for the identified 'val/' metrics
            history = run.history(keys=save_metrics, pandas=True)
            
            # There should be >= 2 rows in the history
            if len(history) < 2:
                print(f"Warning: Run {run.name} has less than 2 rows in the history")
                continue # Skip this run if there are less than 2 rows in the history

            # Add run ID, created_at, and run name
            history['run_id'] = run.id
            history['run_created_at'] = run.created_at
            history['run_name'] = run.name

            run_data[run.id] = {
                'data': history,
                'config': run.config,
                'summary': run.summary._json_dict,
                'created_at': run.created_at,
                'name': run.name
            }

        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
    
    return run_data

def save_data(run_data, output_dir="wandb_data"):
    """Save the extracted data to consolidated files."""
    os.makedirs(output_dir, exist_ok=True)
    
    data_groups = {}
    for run_id, run_info in run_data.items():
        run_name = run_info['name']
        if run_name not in data_groups:
            data_groups[run_name] = []
        data_groups[run_name].append(run_info)

    
    for run_name, run_data in data_groups.items():
        os.makedirs(os.path.join(output_dir, run_name), exist_ok=True)
        group_dir = os.path.join(output_dir, run_name)
        for run_info in run_data:
            run_data = run_info['data']
            run_created_at = run_info['created_at']
            
            run_data.to_csv(os.path.join(group_dir, f'run_data_{run_created_at}.csv'), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Crawl and process W&B data for running experiments.')
    parser.add_argument('--project', type=str, default='Reasoning360', help='Project name')
    parser.add_argument('--entity', type=str, default='leoii22-uc-san-diego', help='Entity name')
    parser.add_argument('--wandb-api-key', type=str, default='633cdb1f1b9dfb2ae2681e47863635fe33b93a10', help='wandb api key')
    parser.add_argument('--run_ids', type=str, default=None, help='run ids')
    parser.add_argument('--output-dir', type=str, default='wandb_data_response_length', help='Directory to save output files')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)

    metrics = ["response_length", "critic/rewards", "critic/scores"]
    run_data = crawl_wandb_data(
        project_name=args.project, 
        entity=args.entity, 
        metric_names=metrics,
        run_ids=args.run_ids.split(",")
    )
    
    # Save combined data to consolidated files
    save_data(run_data, args.output_dir)

    print("Data crawling and processing completed!")