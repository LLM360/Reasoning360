import pandas as pd
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np # Import numpy for handling potential NaN/inf in normalization


training_domain_map = {
    "guru15k_math2.5k": "+Math",
    "guru15k_codegen2.5k": "+Codegen",
    "guru15k_logic2.5k": "+Logic",
    "guru15k_simulation2.5k": "+Simulation",
    "guru15k_table2.5k": "+Table",
    "guru15k_stem2.5k": "+STEM",
    "guru15k_mix": "Mix All",
}

# Define the desired order for task domain groups
task_domain_group_order = ["+Math", "+Codegen", "+Logic", "+Simulation", "+Table", "+STEM"]
task_alias = {
    # Math
    "math": "MATH500",
    "amc_repeated_4x": "AMC",
    "aime_repeated_8x": "AIME24",
    # Codegen
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "livecodebench": "LiveCodeBench",
    # Logic
    "arcagi1": "ARC-AGI",
    "graph_logical_dataset": "GraphLogic",
    "ordering_puzzle_dataset": "OrderLogic",
    "zebra_puzzle_dataset": "ZebraLogic",
    # Simulation
    "codeio": "Code I/O",
    # Structure
    "hitab": "HiTab",
    "multihier": "MultiHiertt",
    # Science
    "gpqa": "GPQA Diamond",
}
disired_column_order = ["MATH500", "AMC", "AIME24", "HumanEval", "MBPP", "LiveCodeBench", "ARC-AGI", "GraphLogic", "OrderLogic", "ZebraLogic", "Code I/O", "HiTab", "MultiHiertt", "GPQA Diamond"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_data_dir", type=str, default="wandb_data")
    args = parser.parse_args()

    # Load and concatenate data for each experiment
    data_dir = args.wandb_data_dir
    exp_dirs = [
        "guru15k_math2.5k",
        "guru15k_codegen2.5k",
        "guru15k_logic2.5k",
        "guru15k_simulation2.5k",
        "guru15k_table2.5k",
        "guru15k_stem2.5k",
        "guru15k_mix",
    ]

    combined_dataframes = {}
    all_best_performance = {}

    for exp_dir in exp_dirs:
        exp_data_dir = os.path.join(data_dir, exp_dir)
        if not os.path.exists(exp_data_dir):
            print(f"Warning: Directory not found: {exp_data_dir}. Skipping.")
            continue

        exp_data_files = os.listdir(exp_data_dir)
        exp_data_files = [f for f in exp_data_files if f.endswith(".csv")]

        if not exp_data_files:
            print(f"Warning: No CSV files found in {exp_data_dir}. Skipping.")
            continue

        # Read and concatenate all CSVs for the current experiment
        list_of_dfs = []
        for csv_file in exp_data_files:
            file_path = os.path.join(exp_data_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                list_of_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        if not list_of_dfs:
            print(f"No data loaded from CSVs in {exp_data_dir}. Skipping.")
            continue

        df_combined = pd.concat(list_of_dfs, ignore_index=True)

        # Sort by 'step'
        df_combined = df_combined.rename(columns={"_step": "step"})
        df_combined = df_combined.sort_values(by='step', ascending=True).reset_index(drop=True)
        combined_dataframes[exp_dir] = df_combined

        # Calculate best performance for each column (except step)
        best_performance = {}
        score_cols = df_combined.columns.drop('step', errors='ignore') # 'errors' handles case where 'step' doesn't exist

        if 'step' not in df_combined.columns:
             print(f"Warning: 'step' column not found in {exp_dir}. Cannot calculate best performance.")
             all_best_performance[exp_dir] = {"error": "'step' column missing"}
             continue

        if score_cols.empty:
            print(f"Warning: No score columns found in {exp_dir}. Cannot calculate best performance.")
            all_best_performance[exp_dir] = {"error": "No score columns found"}
            continue

        for col in score_cols:
            # Ensure the column is numeric before finding max
            if pd.api.types.is_numeric_dtype(df_combined[col]):
                max_score = df_combined[col].max()
                # Find the step corresponding to the maximum score. Use idxmax which returns the first occurrence if there are ties.
                if not pd.isna(max_score): # Only proceed if max_score is not NaN or inf
                    # Handle potential empty dataframe or column with all NaNs case before idxmax
                    if df_combined[col].isnull().all():
                         print(f"Warning: Column '{col}' in {exp_dir} is all NaN. Skipping best performance storage.")
                         continue

                    try:
                        best_step_for_col = df_combined.loc[df_combined[col].idxmax(), 'step']
                        best_performance[f"best_{col}"] = {"step": int(best_step_for_col), "score": float(max_score)}
                    except Exception as e:
                         print(f"Error finding idxmax for column '{col}' in {exp_dir}: {e}. Skipping.")
                else:
                    print(f"Warning: Max score for column '{col}' in {exp_dir} is NaN or Inf ({max_score}). Skipping best performance storage for this column.")
            else:
                print(f"Warning: Column '{col}' in {exp_dir} is not numeric. Skipping best performance calculation for this column.")


        # Calculate the step that delivers the best average score across all benchmarks
        # Use only numeric score columns for average calculation
        numeric_score_cols = df_combined[score_cols].select_dtypes(include=['number']).columns
        if not numeric_score_cols.empty:
            df_combined['avg_score'] = df_combined[numeric_score_cols].mean(axis=1)

            best_avg_score_value = df_combined['avg_score'].max()
            # Find the step for the best average score
            if not pd.isna(best_avg_score_value): # Check for NaN/Inf
                if df_combined['avg_score'].isnull().all():
                     print(f"Warning: Average score column in {exp_dir} is all NaN. Skipping best average score storage.")
                else:
                    try:
                        best_avg_score_step_value = df_combined.loc[df_combined['avg_score'].idxmax(), 'step']

                        best_performance["best_average_score"] = {
                            "step": int(best_avg_score_step_value),
                            "score": float(best_avg_score_value)
                        }
                    except Exception as e:
                         print(f"Error finding idxmax for average score in {exp_dir}: {e}. Skipping.")
            else:
                print(f"Warning: Max average score in {exp_dir} is NaN or Inf ({best_avg_score_value}). Skipping best average score storage.")
        else:
             print(f"Warning: No numeric score columns found in {exp_dir} to calculate average score.")


        all_best_performance[exp_dir] = best_performance

    # Print the results (optional, from previous step)
    print("\n--- Best Performance Summary ---")
    for exp, results in all_best_performance.items():
        print(f"\nExperiment: {exp}")
        if "error" in results:
            print(f"  Error: {results['error']}")
        elif not results:
             print("  No valid performance data found.")
        else:
            for metric, data in results.items():
                # Exclude best_average_score from this detailed print if you only want per-task bests
                if metric == "best_average_score":
                     print(f"  {metric}: Step {data['step']}, Score {data['score']:.4f}")
                elif metric.startswith("best_"):
                     print(f"  {metric}: Step {data['step']}, Score {data['score']:.4f}")


    # --- Heatmap Generation ---
    def extract_domain_task(col):
        # Example col: "val/test_score/math__algebra"
        try:
            print(col)
            domain_task = col.split('/')[2]  # "math__algebra"
            domain, task = domain_task.split('__', 1)
            task = task_alias.get(task, task)
            if task == 'ARC-AGI':
                domain = "Logic"
            return domain, task
        except Exception:
            return None, None

    # Prepare data for heatmap
    heatmap_data = []


    for exp_dir, results in all_best_performance.items():
        if "error" in results or not results:
            continue # Skip experiments that had loading or processing errors

        training_domain_label = training_domain_map.get(exp_dir, exp_dir)

        for metric_key, data in results.items():
            if metric_key.startswith("best_") and metric_key != "best_average_score":
                # Original column name is metric_key after removing "best_"
                original_col_name = metric_key.replace("best_", "")
                # task_domain_group = get_task_domain_group(original_col_name)
                domain, task = extract_domain_task(original_col_name)
                print(domain, task, data['score'])

                if domain != 'other': # Only include columns with identifiable domain groups
                     score = data['score']
                     # Ensure score is a number, skip if not (though previous checks should handle this)
                     if isinstance(score, (int, float)) and not pd.isna(score):
                         heatmap_data.append({
                             "Training Data": training_domain_label,
                             "Task Domain Group": domain,
                             "Task Name": task, # Use original col name for column header
                             "Score": score
                         })
                     else:
                         print(f"Skipping non-numeric or invalid score for {original_col_name} in {exp_dir}")


    # Draw the heatmap
    if not heatmap_data:
        print("No valid data collected for heatmap.")
    else:
        heatmap_df = pd.DataFrame(heatmap_data)

        # Sort dataframe first by Task Domain Group (in defined order) and then by Task Name
        # Use Categorical type to enforce the specific order of domain groups
        heatmap_df['Task Domain Group'] = pd.Categorical(
            heatmap_df['Task Domain Group'],
            categories=task_domain_group_order,
            ordered=True
        )

        # Apply extraction to each row
        print(heatmap_df['Task Name'])
        # heatmap_df[['Domain', 'Task Name']] = heatmap_df['Task Name'].apply(
        #     lambda x: pd.Series(extract_domain_task(x))
        # )
        # heatmap_df = heatmap_df.reindex(columns=disired_column_order)

        # Create the matrix of original scores
        score_matrix = heatmap_df.pivot_table(
            index="Training Data",
            columns="Task Name", # Use individual task names as columns
            values="Score"
        )
        score_matrix = score_matrix.reindex(columns=disired_column_order)

        # Create a normalized matrix for coloring (column-wise min-max scaling)
        normalized_score_matrix = score_matrix.copy()
        epsilon = 1e-9 # Small value to prevent division by zero

        for col in normalized_score_matrix.columns:
            min_val = normalized_score_matrix[col].min()
            max_val = normalized_score_matrix[col].max()
            # Normalize column-wise
            # Check if there's variation in the column
            if (max_val - min_val) > epsilon:
                 normalized_score_matrix[col] = (normalized_score_matrix[col] - min_val) / (max_val - min_val)
            else:
                 # If max and min are the same (constant values), assign a neutral value (e.g., 0.5)
                 normalized_score_matrix[col] = 0.5
            # Handle potential NaN results from normalization if column had NaNs
            normalized_score_matrix[col] = normalized_score_matrix[col].fillna(0) # Treat NaN scores as lowest for coloring? Or leave as NaN? Leaving as NaN might show gray. Let's fill with 0 for consistent coloring scale.

        # Ensure row order matches training_domain_map order
        # Filter row_order to only include training domains that actually have data in the matrix
        row_order = [training_domain_map[d] for d in exp_dirs if training_domain_map[d] in score_matrix.index]
        print(row_order)

        # Reindex matrices to enforce row order
        score_matrix = score_matrix.reindex(index=row_order)
        normalized_score_matrix = normalized_score_matrix.reindex(index=row_order)


        # Define custom blue-to-green colormap
        # Blue for low (0.0), Green for high (1.0)
        colors = ["red", "green"]
        custom_cmap = LinearSegmentedColormap.from_list("blue_to_orange", colors)


        # Plot the heatmap
        plt.figure(figsize=(18, 8)) # Adjust figure size as needed for column labels

        # Use the normalized matrix for coloring and the original matrix for annotations
        # Add edge color and line width for better cell separation
        sns.heatmap(normalized_score_matrix, annot=score_matrix, fmt=".3f", cmap=custom_cmap,
                    linewidths=.5, linecolor='lightgray', cbar_kws={'label': 'Column-wise Score (Normalized)'})

        plt.title("Training on different domains and evaluate on all domains to see how well the domain-data generalizes")
        plt.xlabel("Task Name")
        plt.ylabel("Training Data Domain")
        plt.xticks(rotation=45, ha='right') # Rotate column labels for readability
        plt.yticks(rotation=0) # Keep row labels horizontal
        plt.tight_layout()
        plt.show()
        plt.savefig("heatmap.png")

