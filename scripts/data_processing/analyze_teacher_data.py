"""
Script to analyze teacher response data structure and alignment with training data.
This was used to understand the data before creating the processing pipeline.
"""

import polars as pl
import pyarrow.parquet as pq
import numpy as np


def read_parquet_safe(file_path):
    """Read parquet file with polars."""
    print(f"  Loading with polars...")
    df = pl.read_parquet(file_path)
    print(f"  Loaded successfully: {df.shape}")
    return df


def analyze_schema(file_path, name="Data"):
    """Analyze the schema of a parquet file."""
    print(f"\n{'='*60}")
    print(f"{name.upper()} SCHEMA")
    print(f"{'='*60}")

    schema = pq.read_schema(file_path)
    metadata = pq.read_metadata(file_path)

    print(f"File: {file_path}")
    print(f"Total rows: {metadata.num_rows}")
    print(f"\nColumns ({len(schema.names)}):")
    print(schema.names)

    return schema, metadata


def analyze_sample_data(file_path, columns_to_check, name="Data", n_samples=3):
    """Analyze sample data from a parquet file using polars."""
    print(f"\n{'='*60}")
    print(f"{name.upper()} SAMPLE DATA")
    print(f"{'='*60}")

    # Read with polars
    df_full = read_parquet_safe(file_path)

    # Take first few rows
    df = df_full.head(n_samples)

    # Check which columns exist
    available_cols = [col for col in columns_to_check if col in df.columns]
    missing_cols = [col for col in columns_to_check if col not in df.columns]

    if missing_cols:
        print(f"\nColumns not found in data: {missing_cols}")

    print(f"\nAnalyzing columns: {available_cols}")

    for col in available_cols:
        print(f"\n{col}:")

        # Get first non-null value
        non_null_rows = df.filter(pl.col(col).is_not_null())
        if len(non_null_rows) == 0:
            print(f"  All values are null")
            continue

        sample_val = non_null_rows[col][0]
        print(f"  Type: {type(sample_val)}")

        if isinstance(sample_val, (list, tuple)):
            print(f"  Length: {len(sample_val)}")
            if len(sample_val) > 0:
                first_elem = sample_val[0]
                print(f"  First element type: {type(first_elem)}")
                if isinstance(first_elem, str):
                    print(f"  First element preview (200 chars):")
                    print(f"    {repr(first_elem[:200])}")
                elif isinstance(first_elem, dict):
                    print(f"  First element: {first_elem}")
                else:
                    print(f"  First element: {str(first_elem)[:200]}")
        elif isinstance(sample_val, (str, int, float, bool)):
            print(f"  Value: {sample_val}")
        else:
            print(f"  Preview: {str(sample_val)[:200]}")

    return df


def compare_schemas(teacher_file, train_file):
    """Compare schemas between teacher and training data."""
    print(f"\n{'='*60}")
    print("SCHEMA COMPARISON")
    print(f"{'='*60}")

    teacher_schema = pq.read_schema(teacher_file)
    train_schema = pq.read_schema(train_file)

    teacher_cols = set(teacher_schema.names)
    train_cols = set(train_schema.names)

    common_cols = teacher_cols & train_cols
    teacher_only = teacher_cols - train_cols
    train_only = train_cols - teacher_cols

    print(f"\nCommon columns ({len(common_cols)}):")
    print(sorted(list(common_cols)))

    print(f"\nTeacher-only columns ({len(teacher_only)}):")
    print(sorted(list(teacher_only)))

    print(f"\nTraining-only columns ({len(train_only)}):")
    print(sorted(list(train_only)))

    return common_cols, teacher_only, train_only


def analyze_teacher_responses(teacher_file, n_samples=500):
    """Analyze the teacher response field in detail using polars."""
    print(f"\n{'='*60}")
    print("TEACHER RESPONSE ANALYSIS")
    print(f"{'='*60}")

    # Read full file with polars
    print(f"Reading file (this may take a moment)...")
    df_full = read_parquet_safe(teacher_file)

    # Take sample
    df = df_full.head(n_samples)
    print(f"\nSample size: {len(df)}")

    # Check for teacher responses
    df_with_resp = df.filter(pl.col('r1_0528_responses').is_not_null())
    print(f"Samples with responses: {len(df_with_resp)}/{len(df)}")

    if len(df_with_resp) == 0:
        print("No teacher responses found in sample!")
        return df

    # Analyze response counts - convert to list to work with
    responses_list = df_with_resp['r1_0528_responses'].to_list()
    response_counts = [len(r) if hasattr(r, '__len__') else 0 for r in responses_list]

    print(f"\nResponses per prompt:")
    print(f"  Min: {min(response_counts)}")
    print(f"  Max: {max(response_counts)}")
    print(f"  Mean: {np.mean(response_counts):.2f}")
    if max(response_counts) < 100:  # Only show distribution if reasonable
        print(f"  Distribution: {np.bincount(response_counts).tolist()}")

    # Analyze pass rates if available
    if 'deepseek_r1_0528_pass_rate' in df_with_resp.columns:
        pass_rate_stats = df_with_resp.select(pl.col('deepseek_r1_0528_pass_rate')).drop_nulls()
        if len(pass_rate_stats) > 0:
            print(f"\nTeacher pass rates:")
            print(f"  Min: {pass_rate_stats['deepseek_r1_0528_pass_rate'].min():.4f}")
            print(f"  Max: {pass_rate_stats['deepseek_r1_0528_pass_rate'].max():.4f}")
            print(f"  Mean: {pass_rate_stats['deepseek_r1_0528_pass_rate'].mean():.4f}")
            print(f"  Median: {pass_rate_stats['deepseek_r1_0528_pass_rate'].median():.4f}")

    # Analyze response lengths
    if 'response_lengths' in df_with_resp.columns:
        all_lengths = []
        lengths_list = df_with_resp['response_lengths'].head(100).to_list()
        for lengths in lengths_list:
            if hasattr(lengths, '__len__'):
                all_lengths.extend(list(lengths))

        if all_lengths:
            print(f"\nResponse lengths (characters, from {len(all_lengths)} responses):")
            print(f"  Average: {np.mean(all_lengths):.0f}")
            print(f"  Min: {np.min(all_lengths)}")
            print(f"  Max: {np.max(all_lengths)}")
            print(f"  Median: {np.median(all_lengths):.0f}")
            print(f"  25th percentile: {np.percentile(all_lengths, 25):.0f}")
            print(f"  75th percentile: {np.percentile(all_lengths, 75):.0f}")

    # Show example response
    print(f"\nExample teacher response:")
    first_responses = responses_list[0]
    if hasattr(first_responses, '__len__') and len(first_responses) > 0:
        sample = first_responses[0]
        print(f"  Number of responses for this question: {len(first_responses)}")
        print(f"  Response 1 length: {len(sample)} characters")
        print(f"  First 500 chars:")
        print(sample[:500])

    # Show length statistics if available
    if 'length_statistics' in df_with_resp.columns:
        stats = df_with_resp['length_statistics'][0]
        if stats is not None:
            print(f"\nExample length statistics:")
            print(f"  {stats}")

    return df


def check_alignment(teacher_file, train_file, n_samples=100):
    """Check if data can be aligned using prompt field using polars."""
    print(f"\n{'='*60}")
    print("ALIGNMENT CHECK")
    print(f"{'='*60}")

    # Read samples from both using polars
    print("Reading teacher data sample...")
    df_teacher_full = read_parquet_safe(teacher_file)
    df_teacher = df_teacher_full.head(n_samples)

    print("Reading training data sample...")
    df_train_full = read_parquet_safe(train_file)
    df_train = df_train_full.head(n_samples)

    print(f"\nComparing first {n_samples} samples...")
    print(f"Teacher samples: {len(df_teacher)}")
    print(f"Training samples: {len(df_train)}")

    # Try to match prompts
    def prompt_to_str(p):
        if isinstance(p, list):
            return str(p)
        return str(p)

    print("Creating prompt keys...")
    teacher_prompts_list = df_teacher['prompt'].to_list()
    train_prompts_list = df_train['prompt'].to_list()

    teacher_prompts = set(prompt_to_str(p) for p in teacher_prompts_list)
    train_prompts = set(prompt_to_str(p) for p in train_prompts_list)

    matches = teacher_prompts & train_prompts
    print(f"\nMatching prompts: {len(matches)}/{min(len(teacher_prompts), len(train_prompts))}")
    if min(len(teacher_prompts), len(train_prompts)) > 0:
        print(f"Match rate: {len(matches)/min(len(teacher_prompts), len(train_prompts))*100:.2f}%")

    # Check data sources
    if 'data_source' in df_teacher.columns:
        sources = df_teacher['data_source'].value_counts()
        print(f"\nTeacher data sources: {dict(zip(sources['data_source'].to_list(), sources['count'].to_list()))}")
    if 'data_source' in df_train.columns:
        sources = df_train['data_source'].value_counts()
        print(f"Training data sources: {dict(zip(sources['data_source'].to_list(), sources['count'].to_list()))}")

    # Check if training data already has teacher responses
    if 'r1_0528_responses' in df_train.columns:
        has_responses = df_train.filter(pl.col('r1_0528_responses').is_not_null())
        print(f"\nTraining data already has teacher responses:")
        print(f"  Samples with responses: {len(has_responses)}/{len(df_train)}")
        print(f"  Percentage: {len(has_responses)/len(df_train)*100:.2f}%")


def main():
    """Main analysis function."""
    # File paths
    teacher_file = '/lustrefs/users/haonan.li/data/k2/backup/train_scored_12k_len_1/math__combined_118.2k.part1.parquet'
    train_file = '/lustrefs/users/zhuojun.cheng/vpim/guru_data/train/postprocessed_dedup_am_semantic_filtered_0.05_0.94_thresh_ratio0.5_sample1.0_balanced_step2/math__combined_118.2k.part1_scored.parquet'

    print("="*60)
    print("TEACHER DATA AND TRAINING DATA ANALYSIS")
    print("="*60)

    # 1. Analyze schemas
    teacher_schema, teacher_meta = analyze_schema(teacher_file, "Teacher Data")
    train_schema, train_meta = analyze_schema(train_file, "Training Data")

    # 2. Compare schemas
    common, teacher_only, train_only = compare_schemas(teacher_file, train_file)

    # 3. Analyze sample data from teacher
    teacher_sample_cols = ['data_source', 'prompt', 'r1_0528_responses',
                           'deepseek_r1_0528_pass_rate', 'response_lengths']
    analyze_sample_data(teacher_file, teacher_sample_cols, "Teacher Data")

    # 4. Analyze sample data from training
    train_sample_cols = ['data_source', 'prompt', 'r1_0528_responses']
    analyze_sample_data(train_file, train_sample_cols, "Training Data")

    # 5. Detailed analysis of teacher responses
    analyze_teacher_responses(teacher_file)

    # 6. Check alignment possibility
    check_alignment(teacher_file, train_file, n_samples=100)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey findings:")
    print("1. Teacher data has r1_0528_responses with multiple responses per prompt")
    print("2. Training data has student performance metrics (pass_rate, model_pass_rate)")
    print("3. Data can be aligned using 'prompt' field")
    print("4. Row counts differ - training data has more samples")
    print("\nNext step: Create data processing script to merge datasets")


if __name__ == '__main__':
    main()
