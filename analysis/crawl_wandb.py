import wandb
import pandas as pd
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def crawl_wandb_data(project_name, entity, metric_names, run_name_filter=None, run_name_list=None):
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
    api = wandb.Api()
    
    # Convert single metric to list for consistent handling
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    # Get all runs from the project that are in running state
    # Note: We're not filtering by run_name_filter to keep all running experiments
    project_path = f"{entity}/{project_name}" if entity else project_name
    # Construct filters based on input
    # filters = {}
    # if run_name_list:
    #     print(f"Filtering runs by name: {run_name_list}")
    #     # filters["name"] = {"$in": run_name_list}
    #     filter
    # elif run_name_filter:
    #     print(f"Filtering runs by name: {run_name_filter}")
    #     # Basic substring matching for run_name_filter - adjust if more complex filtering is needed
    #     filters["name"] = {"$regex": run_name_filter}
    # # Get runs from the project based on filters
    # try:
    #     runs = api.runs(project_path, filters=filters)
    # except Exception as e:
    #     runs = api.runs(project_path)
    #     print(f"Error fetching runs for project {project_path} with filters {filters}: {e}")
    #     return {}    
    # print(f"Found {len(runs)} running and finished experiments")
    
    # Extract data for each run
    run_data = {}
    
    # keep only running or finished runs    
    filters = {"state": {"$in": ["running", "finished"]}}
    # run_name_list = ["yolorun-azure-hpc-H200-instance-019.core42.ai-20250512_015607-Qwen2.5-7B-think-4node-guru-full-minibsz64"]
    # if run_name_list: # If a list of names was provided via --run-name-list
    #     filters["name"] = {"$in": run_name_list} # Use the $in filter to match any name in the list
    # runs = api.runs(project_path, filters=filters)
    # Only keep runs finished in two weeks
    # runs = api.runs(project_path, filters=filters)
    runs = api.runs(project_path)
    print(f"Found {len(runs)} runs")
    for run in tqdm(runs):
        if run.name not in run_name_list:
            continue
        try:
            print(f"Processing run {run.name}")

            # Get a sample history to identify available keys
            # Fetching just one step is enough to get column names
            sample_history = run.history(samples=2)
            print(f"Sample history: {sample_history}")

            # Identify metrics that start with "val/"
            print(f"Sample history columns: {sample_history.columns}")
            val_metrics = [key for key in sample_history.columns if key.startswith("val/")]

            if not val_metrics:
                print(f"Warning: No metrics starting with 'val/' found in run {run.name}")
                continue # Skip this run if no 'val/' metrics are found

            print(f"Found 'val/' metrics in run {run.name}: {val_metrics}")

            # Get the full history for the identified 'val/' metrics
            history = run.history(keys=val_metrics, pandas=True)
            
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

def save_data(run_data, output_dir="wandb_data", group_by=None, group_by_alias=None):
    """Save the extracted data to consolidated files."""
    os.makedirs(output_dir, exist_ok=True)
    
    group_by_map = {k: v for k, v in zip(group_by, group_by_alias)}
    
    data_groups = {}
    if group_by:
        for group_name in group_by:
            for run_id, run_info in run_data.items():
                run_name = run_info['name']
                print(f"Run {run_name} is in group {group_name}")
                if group_name in run_name:
                    if group_name not in data_groups:
                        data_groups[group_name] = []
                    data_groups[group_name].append(run_info)

    
    for group_name, group_data in data_groups.items():
        os.makedirs(os.path.join(output_dir, group_by_map[group_name]), exist_ok=True)
        group_dir = os.path.join(output_dir, group_by_map[group_name])
        for run_idx, run_info in enumerate(group_data, 1):
            run_name = run_info['name']
            run_data = run_info['data']
            run_config = run_info['config']
            run_summary = run_info['summary']
            run_created_at = run_info['created_at']
            
            run_data.to_csv(os.path.join(group_dir, f'run_data_{run_created_at}.csv'), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Crawl and process W&B data for running experiments.')
    parser.add_argument('--project', type=str, default='Reasoning360', help='Project name')
    parser.add_argument('--entity', type=str, default='leoii22-uc-san-diego', help='Entity name')
    parser.add_argument('--wandb-api-key', type=str, default='633cdb1f1b9dfb2ae2681e47863635fe33b93a10', help='wandb api key')
    parser.add_argument('--run-name-filter', type=str, default=None, help='Filter to apply to run names')
    parser.add_argument('--run-name-list', nargs='+', default=None, help='List of run names to crawl')
    parser.add_argument('--group-by', nargs='+', default=None, help='Group by run names')
    parser.add_argument('--group-by-alias', nargs='+', default=None, help='Alias for group by run names')
    parser.add_argument('--output-dir', type=str, default='wandb_data', help='Directory to save output files')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
        
    if args.run_name_list:
        print(f"Crawling runs: {args.run_name_list}")
        print(f"Type of run_name_list: {type(args.run_name_list)}")
    
    metrics = ["val/"]
    run_data = crawl_wandb_data(
        project_name=args.project, 
        entity=args.entity, 
        metric_names=metrics,
        run_name_filter=args.run_name_filter,
        run_name_list=args.run_name_list
    )
    
    # Save combined data to consolidated files
    save_data(run_data, args.output_dir, args.group_by, args.group_by_alias)

    print("Data crawling and processing completed!")