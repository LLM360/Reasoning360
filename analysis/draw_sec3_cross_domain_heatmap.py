import pandas as pd
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# --- Configuration ---
GURU15K_TRAINING_DOMAIN_MAP = {
    "guru15k_math2.5k": "+Math",
    "guru15k_codegen2.5k": "+Codegen",
    "guru15k_logic2.5k": "+Logic",
    "guru15k_simulation2.5k": "+Simulation",
    "guru15k_table2.5k": "+Table",
    "guru15k_stem2.5k": "+Science",
    "guru15k_mix": "Mix All",
}
GURU18K_TRAINING_DOMAIN_MAP = {
    "guru18k_math3k": "+Math",
    "guru18k_codegen3k": "+Coding",
    "guru18k_logic3k": "+Logic",
    "guru18k_simulation3k": "+Simulation",
    "guru18k_table3k": "+Table",
    "guru18k_stem3k": "+Science",
    "guru18k_mix": "Mix All",
}

# Original fine-grained task aliases
# COLUMNS_TO_EXCLUDE = ['GraphLogic', "GPQA Diamond"]
COLUMNS_TO_EXCLUDE = ['val/test_score/logic__graph_logical_dataset/unknown',
                      'val/test_score/stem__gpqa_diamond/Idavidrein/gpqa']

TASK_ALIAS = {
    "math": "MATH500", "amc_repeated_4x": "AMC", "aime_repeated_8x": "AIME24",
    "humaneval": "HumanEval", "mbpp": "MBPP", "livecodebench": "LiveCodeBench",
    "arcagi1": "ARC-AGI",
    "ordering_puzzle_dataset": "OrderLogic", "zebra_puzzle_dataset": "ZebraLogic",
    "codeio": "Code I/O",
    "hitab": "HiTab", "multihier": "MultiHiertt",
    # "gpqa_diamond": "GPQA Diamond",
    "supergpqa": "SuperGPQA",
}

# Order for fine-grained evaluation tasks
DETAILED_EVAL_COLUMN_ORDER = [
    "MATH500", "AMC", "AIME24", "HumanEval", "MBPP", "LiveCodeBench",
    "SuperGPQA",
    "ARC-AGI", "OrderLogic", "ZebraLogic", "Code I/O",
    "HiTab", "MultiHiertt", 
]

# Mapping of fine-grained evaluation tasks to their broader domains
EVAL_TASK_TO_DOMAIN_GROUP_MAP = {
    "MATH500": "Math", "AMC": "Math", "AIME24": "Math",
    "HumanEval": "Codegen", "MBPP": "Codegen", "LiveCodeBench": "Codegen",
    "ARC-AGI": "Logic", "OrderLogic": "Logic", "ZebraLogic": "Logic",
    "Code I/O": "Simulation",
    "HiTab": "Table", "MultiHiertt": "Table", # Assuming 'Structure' can be grouped under 'Table' or a new 'Structure' group
     "SuperGPQA": "Science"
}

# Desired order for grouped evaluation domain columns
GROUPED_EVAL_COLUMN_ORDER = ["Math", "Codegen", "Science", "Logic", "Simulation", "Table", ]

GURU15K_EXPERIMENT_DIRS = [
    "guru15k_math2.5k", "guru15k_codegen2.5k", "guru15k_logic2.5k",
    "guru15k_simulation2.5k", "guru15k_table2.5k", "guru15k_stem2.5k",
    "guru15k_mix",
]
GURU18K_EXPERIMENT_DIRS = [
    "guru18k_math3k", "guru18k_codegen3k", "guru18k_stem3k", "guru18k_logic3k",
    "guru18k_simulation3k", "guru18k_table3k", 
    "guru18k_mix",
]


# --- Data Loading and Preprocessing Functions ---
def load_experiment_data(data_dir, exp_dir):
    exp_data_dir = os.path.join(data_dir, exp_dir)
    if not os.path.exists(exp_data_dir):
        # print(f"Warning: Directory not found: {exp_data_dir}. Skipping.")
        return None
    exp_data_files = [f for f in os.listdir(exp_data_dir) if f.endswith(".csv")]
    if not exp_data_files:
        # print(f"Warning: No CSV files found in {exp_data_dir}. Skipping.")
        return None
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
        # print(f"No data loaded from CSVs in {exp_data_dir}. Skipping.")
        return None
    df_combined = pd.concat(list_of_dfs, ignore_index=True)
    df_combined = df_combined.rename(columns={"_step": "step"})
    df_combined = df_combined.sort_values(by='step', ascending=True).reset_index(drop=True)
    return df_combined

def calculate_best_performance_for_experiment(df_combined, exp_dir, use_last_step=False, use_best_avg_step=False):
    if df_combined is None: return {"error": "No data provided"}
    if 'step' not in df_combined.columns: return {"error": "'step' column missing"}

    best_performance = {}
    score_cols = df_combined.columns.drop('step', errors='ignore')
    if score_cols.empty: return {"error": "No score columns found"}

    # Get numeric score columns for average calculation
    numeric_score_cols = df_combined[score_cols].select_dtypes(include=np.number).columns
    
    if use_last_step:
        # Use the last step for all metrics
        last_step = df_combined['step'].max()
        last_step_row = df_combined[df_combined['step'] == last_step].iloc[0]
        for col in numeric_score_cols:
            score = float(last_step_row[col])
            if not pd.isna(score) and np.isfinite(score):
                best_performance[f"best_{col}"] = {"step": int(last_step), "score": score}
        
        # Calculate average score for the last step
        avg_score = last_step_row[numeric_score_cols].mean()
        if not pd.isna(avg_score):
            best_performance["best_average_score"] = {
                "step": int(last_step),
                "score": float(avg_score)
            }
    
    elif use_best_avg_step:
        # Calculate average score at each step
        avg_scores = df_combined[numeric_score_cols].mean(axis=1, skipna=True)
        best_avg_step = df_combined.loc[avg_scores.idxmax(), 'step']
        best_avg_row = df_combined[df_combined['step'] == best_avg_step].iloc[0]
        
        # Use scores from the step with best average for all metrics
        for col in numeric_score_cols:
            score = float(best_avg_row[col])
            if not pd.isna(score) and np.isfinite(score):
                best_performance[f"best_{col}"] = {"step": int(best_avg_step), "score": score}
        
        # Include the best average score
        best_performance["best_average_score"] = {
            "step": int(best_avg_step),
            "score": float(avg_scores.max())
        }
    
    else:
        # Original behavior: find best score for each metric independently
        for col in numeric_score_cols:
            numeric_col = pd.to_numeric(df_combined[col], errors='coerce')
            max_score = numeric_col.max()
            if not pd.isna(max_score):
                if numeric_col.isnull().all(): continue
                try:
                    best_step_for_col = df_combined.loc[numeric_col.idxmax(), 'step']
                    best_performance[f"best_{col}"] = {"step": int(best_step_for_col), "score": float(max_score)}
                except Exception: pass
        
        # Calculate best average score
        avg_scores = df_combined[numeric_score_cols].mean(axis=1, skipna=True)
        best_avg_score = avg_scores.max()
        if not pd.isna(best_avg_score):
            try:
                best_avg_step = df_combined.loc[avg_scores.idxmax(), 'step']
                best_performance["best_average_score"] = {
                    "step": int(best_avg_step),
                    "score": float(best_avg_score)
                }
            except Exception: pass

    return best_performance

def process_all_experiments(data_dir, exp_dirs, use_last_step=False, use_best_avg_step=False):
    all_best_performance = {}
    # Store loaded dataframes to avoid reloading for baseline
    loaded_dfs = {}
    for exp_dir in exp_dirs:
        df_combined = load_experiment_data(data_dir, exp_dir)
        # Drop columns that are not needed for evaluation
        for task_to_exclude in COLUMNS_TO_EXCLUDE:
            print(f"To drop column: {task_to_exclude}")
            print(f"df_combined.columns: {df_combined.columns}")
            if task_to_exclude in df_combined.columns:
                df_combined = df_combined.drop(columns=[task_to_exclude])
        loaded_dfs[exp_dir] = df_combined # Store the loaded df
        all_best_performance[exp_dir] = calculate_best_performance_for_experiment(
            df_combined, exp_dir, use_last_step, use_best_avg_step
        ) if df_combined is not None else {"error": f"Failed to load data for {exp_dir}"}
    return all_best_performance, loaded_dfs

def get_average_baseline_scores(loaded_dfs, task_alias_map):
    """Extracts and averages step=0 scores from loaded dataframes."""
    step0_rows = []
    for exp_dir, df in loaded_dfs.items():
        if df is not None and 0 in df['step'].values:
            step0_row = df[df['step'] == 0].iloc[0] # Get the first (and usually only) step 0 row
            step0_rows.append(step0_row)

    if not step0_rows:
        print("Warning: No step=0 data found in any loaded experiment runs.")
        return {} # Return empty dict if no baseline data

    # Convert list of Series (rows) to a DataFrame
    combined_step0 = pd.DataFrame(step0_rows)

    # Identify potential evaluation score columns (should start with 'evaluation/')
    potential_score_cols = [col for col in combined_step0.columns if col.startswith('val/')]

    # Select only numeric columns from the potential score columns
    numeric_score_cols = combined_step0[potential_score_cols].select_dtypes(include=np.number).columns

    if numeric_score_cols.empty:
        print("Warning: No numeric evaluation score columns found among potential step=0 columns.")
        return {}

    # Calculate the mean *only* for the numeric score columns
    average_scores = combined_step0[numeric_score_cols].mean(axis=0)

    baseline_scores_dict = {}
    for col in numeric_score_cols: # Iterate only over numeric score columns
        # Extract task name using the existing mapping logic
        _, task_name = extract_domain_task_from_col(col, task_alias_map)
        # Only include valid task names (from TASK_ALIAS values) and non-NaN average scores
        if task_name and pd.notna(average_scores[col]) and task_name in TASK_ALIAS.values():
             baseline_scores_dict[task_name] = float(average_scores[col]) # Ensure float type
        elif task_name is None:
             # This warning might be noisy if many columns don't match the format
             # print(f"Debug: Could not extract task name from baseline column '{col}' using split logic. Skipping.")
             pass # Skip silently if format doesn't match
        elif task_name not in TASK_ALIAS.values():
             # This warning is useful if a column matches the format but the alias is unknown
             print(f"Warning: Baseline task alias '{task_name}' derived from column '{col}' is not in TASK_ALIAS. Skipping.")


    return baseline_scores_dict


def print_best_performance_summary(all_best_performance):
    print("\n--- Best Performance Summary (Overall Average) ---")
    for exp, results in all_best_performance.items():
        train_label = GURU18K_TRAINING_DOMAIN_MAP.get(exp, exp)
        if "best_average_score" in results and isinstance(results["best_average_score"], dict):
            data = results["best_average_score"]
            print(f"  Training: {train_label:<15} Best Avg Score: {data['score']:.4f} (at step {data['step']})")
        elif "error" in results:
            print(f"  Training: {train_label:<15} Error: {results['error']}")
        else:
            print(f"  Training: {train_label:<15} No valid average performance data.")

def extract_domain_task_from_col(col_name, task_alias_map):
    try:
        parts = col_name.split('/')
        if len(parts) > 2:
            domain_task_str = parts[2]
            domain_parts = domain_task_str.split('__', 1)
            task_key = domain_parts[1] if len(domain_parts) > 1 else domain_parts[0]
            task_name = task_alias_map.get(task_key, task_key) # Use get for safety
            # Ensure the task_name is one we actually care about (in TASK_ALIAS values)
            if task_name in task_alias_map.values():
                 return domain_parts[0], task_name # Domain here is the one from column name, not the group
    except Exception:
        return None, None
    return None, None # Ensure it always returns a tuple

def prepare_heatmap_data(all_best_performance, baseline_scores_dict, training_domain_map, task_alias_map,
                         group_eval_domains=False, eval_task_to_domain_group_map=None):
    heatmap_data_list = []
    
    # Add Baseline data first
    if baseline_scores_dict:
        for task_name, score in baseline_scores_dict.items():
             record = {
                "Training Data": "Base (Qwen2.5-7B)",
                "Score": score
             }
             if group_eval_domains:
                domain_group = eval_task_to_domain_group_map.get(task_name)
                if domain_group: # Only include tasks that map to a domain group
                    record["Eval Domain Group"] = domain_group
                    heatmap_data_list.append(record)
             else:
                record["Task Name"] = task_name
                heatmap_data_list.append(record)

    # Add Post-training data
    for exp_dir, results in all_best_performance.items():
        if "error" in results or not results: continue
        training_domain_label = training_domain_map.get(exp_dir, exp_dir)

        for metric_key, data in results.items():
            if metric_key.startswith("best_") and metric_key != "best_average_score":
                original_col_name = metric_key.replace("best_", "")
                _, task_name = extract_domain_task_from_col(original_col_name, task_alias_map)

                if task_name:
                    score = data.get('score')
                    if isinstance(score, (int, float)) and not pd.isna(score) and np.isfinite(score):
                        record = {
                            "Training Data": training_domain_label,
                            "Score": score
                        }
                        if group_eval_domains:
                            domain_group = eval_task_to_domain_group_map.get(task_name)
                            if domain_group: # Only include tasks that map to a domain group
                                record["Eval Domain Group"] = domain_group
                                heatmap_data_list.append(record)
                        else:
                            record["Task Name"] = task_name
                            heatmap_data_list.append(record)

    if not heatmap_data_list: return None

    df = pd.DataFrame(heatmap_data_list)
    if group_eval_domains:
        if "Eval Domain Group" not in df.columns:
             print("Warning: 'Eval Domain Group' column missing after preparing data for grouping. Check mappings.")
             return pd.DataFrame() # Return empty if grouping failed
        # Aggregate by averaging scores within each training data and eval domain group
        # Grouping needs to handle the Baseline row correctly - it will be treated as a group of size 1
        df_grouped = df.groupby(["Training Data", "Eval Domain Group"])["Score"].mean().reset_index()
        return df_grouped
    return df

def normalize_score_matrix(score_matrix):
    """
    Normalizes each column of a DataFrame by subtracting the first row's value
    (baseline) and then scaling the resulting delta to the range [-1, 1].

    Original NaN values are filled with a placeholder (-2) in the output.

    Args:
        score_matrix (pd.DataFrame): The input DataFrame of scores.

    Returns:
        pd.DataFrame: The normalized DataFrame with deltas scaled to [-1, 1].
                      Original NaNs are replaced by -2.
    """
    normalized_matrix = score_matrix.astype(float).copy() # Ensure float for calculations
    output_matrix = pd.DataFrame(index=score_matrix.index, columns=score_matrix.columns, dtype=float)
    placeholder_for_nan = -2.0 # Value to replace original NaNs with

    for col in normalized_matrix.columns:
        col_data = normalized_matrix[col]
        
        # Get the baseline value from the first row, ignoring potential NaN
        # If the first value is NaN, we cannot establish a baseline for this column
        baseline = col_data.iloc[0]

        if pd.isna(baseline):
            # If baseline is NaN, we cannot normalize this column meaningfully
            print(f"Warning: Baseline (first row) for column '{col}' is NaN. Filling column with placeholder.")
            output_matrix[col] = placeholder_for_nan
            continue

        # Calculate the delta from the baseline for all values
        delta_data = col_data - baseline

        # Identify where the original NaNs were
        original_nan_mask = col_data.isna()

        # Get non-NaN deltas for min/max calculation
        valid_deltas = delta_data.dropna()

        if valid_deltas.empty:
             # If all values were NaN (except possibly the first row NaN case handled above)
             output_matrix[col] = placeholder_for_nan
             continue
             
        min_delta = valid_deltas.min()
        max_delta = valid_deltas.max()

        epsilon = 1e-9 # Tolerance for floating point comparison

        if abs(max_delta - min_delta) < epsilon:
            # If all valid deltas are the same (constant column after baseline subtraction)
            # Normalize to 0 if the delta is 0, otherwise handle as a constant non-zero delta
            if abs(min_delta) < epsilon:
                 # Delta is effectively 0 everywhere (e.g., original column was constant)
                 output_matrix[col] = 0.0
            else:
                 # All non-NaN deltas are the same but non-zero.
                 # This case is tricky for [-1, 1] normalization based on range.
                 # A reasonable approach is to map this single non-zero delta to 0,
                 # or perhaps sign(delta). Let's map to 0 as it's a constant 'change'
                 # relative to the baseline across non-NaN values.
                 print(f"Warning: All valid deltas in column '{col}' are constant and non-zero after baseline subtraction. Mapping non-NaN deltas to 0.")
                 output_matrix[col] = 0.0
        else:
            # Normalize the non-NaN deltas to the range [-1, 1]
            # Formula: 2 * (x - min) / (max - min) - 1
            # output_matrix[col] = 2 * (delta_data - min_delta) / (max_delta - min_delta) - 1
            output_matrix[col] = (delta_data - min_delta) / (max_delta - min_delta)
            
        # Fill the locations of original NaNs with the placeholder value
        output_matrix.loc[original_nan_mask, col] = placeholder_for_nan

    return output_matrix

# def normalize_score_matrix(score_matrix):
#     normalized_matrix = score_matrix.astype(float).copy() # Ensure float for calculations
#     epsilon = 1e-9
#     for col in normalized_matrix.columns:
#         col_data = normalized_matrix[col].dropna()
#         if col_data.empty:
#             normalized_matrix[col] = 0.0
#             continue
#         min_val, max_val = col_data.min(), col_data.max()
#         if (max_val - min_val) > epsilon:
#             normalized_matrix[col] = (normalized_matrix[col] - min_val) / (max_val - min_val)
#         else:
#             # Normalize constant columns to 1 if max > 0, else 0.
#             # If all values are the same, they all get the same normalized score.
#             # If that value is the max/min across the whole column, it's 1 or 0.
#             normalized_matrix[col] = 1.0 if max_val > 0 else 0.0
#         # Fill original NaNs with a distinct low value for viz, outside the [0, 1] range
#         normalized_matrix[col] = normalized_matrix[col].fillna(-0.05)
#     return normalized_matrix


# --- Plotting ---
def plot_heatmap(score_matrix, normalized_score_matrix, filename="heatmap.png", title_suffix="", hide_numbers=False, group_eval_domains=False):
    if score_matrix.empty:
        print("Score matrix is empty. Cannot generate heatmap.")
        return

    # --- Colormap Setting ---
    # cmap = sns.color_palette("YlGnBu", as_cmap=True)
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    num_rows = len(score_matrix.index)
    num_cols = len(score_matrix.columns)
    # Adjust figure size to maintain square cells
    cell_size = 1.2  # Base size for each cell
    fig_width = max(12, num_cols * cell_size)
    fig_height = max(8, num_rows * cell_size)
    plt.figure(figsize=(fig_width, fig_height))

    # Prepare annotations based on hide_numbers flag
    if hide_numbers:
        annot_setting = False
        fmt_setting = None # fmt is ignored if annot is False
    else:
        # Prepare annotations: format original scores, use "N/A" for NaNs
        annot_labels = score_matrix.applymap(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        annot_setting = annot_labels
        fmt_setting = "s" # Use "s" because annot_labels are strings

    ax = sns.heatmap(
        normalized_score_matrix,
        annot=annot_setting, # Use the determined annot setting
        fmt=fmt_setting, # Use the determined fmt setting
        cmap=cmap,
        linewidths=0.7,
        linecolor='white',
        cbar_kws={'label': 'Column-wise Score (Normalized Min-Max)', 'shrink': 0.75},
        square=True,  # Set square=True to ensure square cells
        robust=False
    )

    # Manually set annotation colors for better contrast ONLY if annotations are shown
    if not hide_numbers:
        flat_norm_scores = normalized_score_matrix.values.flatten(order='C') # Row-major
        flat_original_scores = score_matrix.values.flatten(order='C') # Row-major

        for i, text_obj in enumerate(ax.texts):
            original_score_was_nan = pd.isna(flat_original_scores[i])

            if original_score_was_nan:
                 text_obj.set_color('darkgrey') # Color for "N/A" text
                 text_obj.set_fontsize(8)
            else:
                norm_val = flat_norm_scores[i]
                # Adjust threshold based on the chosen colormap (YlGnBu goes from light to dark)
                if norm_val > 0.55: # Threshold for dark background cells
                    text_obj.set_color('white')
                else:
                    text_obj.set_color('black')
                text_obj.set_fontsize(9 if num_cols < 10 else 8)


    # plt.title(f"Generalization of Training Data {title_suffix}", fontsize=16, pad=20, weight='bold')
    if group_eval_domains:
        plt.xlabel("Evaluation Task (Avg. across Domain Group)", fontsize=12, labelpad=12)
    else:
        plt.xlabel("Evaluation Task", fontsize=12, labelpad=15)
    plt.ylabel("Training Data Domain", fontsize=12, labelpad=1.5)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    ax.tick_params(axis='both', which='major', pad=6)
    plt.tight_layout(pad=2.0)

    try:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Heatmap saved to {filename}")
        # plt.show()
    except Exception as e:
        print(f"Error saving heatmap: {e}")

def save_score_matrix(score_matrix, output_filename):
    # Save the score matrix to a CSV file
    score_matrix.to_csv(output_filename, index=True)
    print(f"Score matrix saved to {output_filename}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Process WandB data and generate a performance heatmap.")
    parser.add_argument("--wandb_data_dir", type=str, default="wandb_data",
                        help="Directory containing the WandB experiment data.")
    parser.add_argument("--output_filename", type=str, default="domain_generalization_heatmap.png",
                        help="Base name for the output heatmap file (will append _grouped if grouping).")
    parser.add_argument("--group_eval_domains", action='store_true',
                        help="Group evaluation tasks by broader domains and average scores.")
    parser.add_argument("--hide-numbers", action='store_true',
                        help="Hide the numerical scores within the heatmap cells.")
    parser.add_argument("--use-last-step", action='store_true',
                        help="Use the performance from the last step for all metrics (except baseline).")
    parser.add_argument("--use-best-avg-step", action='store_true',
                        help="Use the performance from the step with best average score for all metrics (except baseline).")
    args = parser.parse_args()

    if args.use_last_step and args.use_best_avg_step:
        print("Error: Cannot use both --use-last-step and --use-best-avg-step at the same time.")
        return

    # Process all experiment data to get best performance and loaded dataframes
    all_best_performance, loaded_dfs = process_all_experiments(
        args.wandb_data_dir, 
        GURU18K_EXPERIMENT_DIRS,
        use_last_step=args.use_last_step,
        use_best_avg_step=args.use_best_avg_step
    )
    print_best_performance_summary(all_best_performance)

    # Modify output filename based on step selection mode
    if args.use_last_step:
        args.output_filename = args.output_filename.replace(".png", "_last_step.png")
    elif args.use_best_avg_step:
        args.output_filename = args.output_filename.replace(".png", "_best_avg_step.png")

    # Modify output filename based on step selection mode
    if args.group_eval_domains:
        current_column_order = GROUPED_EVAL_COLUMN_ORDER
        title_suffix = "across Evaluation Domain Groups"
        output_filename = args.output_filename.replace(".png", "_grouped.png")
    else:
        current_column_order = DETAILED_EVAL_COLUMN_ORDER
        title_suffix = "across Evaluation Tasks"
        output_filename = args.output_filename.replace(".png", "_detailed.png")
    if args.hide_numbers:
        output_filename = output_filename.replace(".png", "_hide_numbers.png")

    # Get baseline performance from step=0 across all loaded runs
    baseline_scores_dict = get_average_baseline_scores(loaded_dfs, TASK_ALIAS)
    if baseline_scores_dict:
        print("\n--- Base (Qwen2.5-7B) Performance (Average at Step 0) ---")
        for task, score in baseline_scores_dict.items():
            print(f"  {task:<15}: {score:.4f}")
        print("-" * 40)


    # Prepare data for heatmap, including baseline
    heatmap_df = prepare_heatmap_data(
        all_best_performance,
        baseline_scores_dict, # Pass baseline data
        GURU18K_TRAINING_DOMAIN_MAP,
        TASK_ALIAS,
        args.group_eval_domains,
        EVAL_TASK_TO_DOMAIN_GROUP_MAP
    )

    if heatmap_df is None or heatmap_df.empty:
        print("No data available for heatmap generation. Exiting.")
        return

    pivot_column = "Eval Domain Group" if args.group_eval_domains else "Task Name"
    try:
        score_matrix = heatmap_df.pivot_table(
            index="Training Data",
            columns=pivot_column,
            values="Score"
        )
    except Exception as e:
        print(f"Error creating pivot table using column '{pivot_column}': {e}")
        print("Columns in heatmap_df:", heatmap_df.columns)
        print("Sample of heatmap_df:\n", heatmap_df.head())
        return

    # Define the desired row order, adding the baseline at the top
    desired_row_order = ["Base (Qwen2.5-7B)"] + \
                        [GURU18K_TRAINING_DOMAIN_MAP[d] for d in GURU18K_EXPERIMENT_DIRS if GURU18K_TRAINING_DOMAIN_MAP[d] in score_matrix.index]
    # Filter out any desired rows not actually present in the data
    desired_row_order = [row for row in desired_row_order if row in score_matrix.index]
        

    # Add missing desired columns with NaN
    for col in current_column_order:
        if col not in score_matrix.columns:
            score_matrix[col] = np.nan

    # Reindex to ensure desired order and inclusion of all specified columns/rows
    # Filter current_column_order to only include columns actually present OR added as NaN
    final_column_order = [col for col in current_column_order if col in score_matrix.columns]
    score_matrix = score_matrix.reindex(index=desired_row_order, columns=final_column_order)
    
    save_score_matrix(score_matrix, output_filename.replace(".png", "_score_matrix.csv"))
    # try:
    #     print("score_matrix.columns:", score_matrix.columns)
    #     score_matrix = score_matrix.drop(columns=COLUMNS_TO_EXCLUDE)
    # except Exception as e:
    #     print(f"Error dropping columns: {e}")
    #     print("Columns in score_matrix:", score_matrix.columns)
    #     print("Sample of score_matrix:\n", score_matrix.head())

    # Normalize the matrix for coloring (excluding baseline from normalization range if needed,
    # but column-wise normalization should handle this implicitly as baseline is just another row)
    normalized_score_matrix = normalize_score_matrix(score_matrix.copy())

    # Pass the hide_numbers flag to the plotting function
    plot_heatmap(score_matrix, normalized_score_matrix,
                 filename=output_filename,
                 title_suffix=title_suffix,
                 hide_numbers=args.hide_numbers,
                 group_eval_domains=args.group_eval_domains) # Pass the new argument

if __name__ == "__main__":
    main()