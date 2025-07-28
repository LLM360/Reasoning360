#!/usr/bin/env python3
"""
Check and fix IFBench data format.
"""

import pandas as pd
import json

def check_ifbench_data(file_path):
    """Check the format of IFBench data file."""
    print(f"Checking file: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        print(f"Total rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Check data structure
        print("\nSample data:")
        print(df.head(2))
        
        # Check for None values in extra_info
        if 'extra_info' in df.columns:
            none_count = df['extra_info'].isna().sum()
            print(f"\nNone values in extra_info: {none_count}")
            
            # Show sample of None values
            if none_count > 0:
                print("\nSample rows with None extra_info:")
                print(df[df['extra_info'].isna()].head(2))
        
        # Check prompt structure
        if 'prompt' in df.columns:
            print(f"\nPrompt column type: {df['prompt'].dtype}")
            print("Sample prompt:")
            print(df['prompt'].iloc[0])
        
        # Check reward_model structure
        if 'reward_model' in df.columns:
            print(f"\nReward model column type: {df['reward_model'].dtype}")
            print("Sample reward_model:")
            print(df['reward_model'].iloc[0])
            
    except Exception as e:
        print(f"Error reading file: {e}")

def fix_ifbench_data(input_file, output_file):
    """Fix IFBench data by ensuring extra_info is not None."""
    print(f"Fixing data: {input_file} -> {output_file}")
    
    df = pd.read_parquet(input_file)
    
    # Fix None values in extra_info
    if 'extra_info' in df.columns:
        # Replace None with dict containing default fields
        def fix_extra_info(info):
            if info is None:
                return {"split": "train", "instruction_id_list": []}
            elif isinstance(info, str):
                try:
                    parsed = json.loads(info)
                    if not parsed:  # Empty dict
                        return {"split": "train", "instruction_id_list": []}
                    return parsed
                except:
                    return {"split": "train", "instruction_id_list": []}
            elif isinstance(info, dict):
                if not info:  # Empty dict
                    return {"split": "train", "instruction_id_list": []}
                return info
            else:
                return {"split": "train", "instruction_id_list": []}
        
        df['extra_info'] = df['extra_info'].apply(fix_extra_info)
    
    # Ensure data_source exists
    if 'data_source' not in df.columns:
        df['data_source'] = 'ood__ifbench'
    
    # Save fixed data
    df.to_parquet(output_file, index=False)
    print(f"Fixed data saved to: {output_file}")
    
    # Verify the fix
    check_ifbench_data(output_file)

def main():
    # Check original data
    original_file = "/mnt/sharefs/users/jianshu.she/ood__ifbench_95.1k.parquet"
    print("=== Checking original data ===")
    check_ifbench_data(original_file)
    
    # Check split data if exists
    train_file = "/mnt/sharefs/users/jianshu.she/ifbench_split/ifbench_train.parquet"
    test_file = "/mnt/sharefs/users/jianshu.she/ifbench_split/ifbench_test.parquet"
    
    import os
    if os.path.exists(train_file):
        print("\n=== Checking train data ===")
        check_ifbench_data(train_file)
    
    if os.path.exists(test_file):
        print("\n=== Checking test data ===")
        check_ifbench_data(test_file)
    
    # Fix the data
    print("\n=== Fixing data ===")
    fix_ifbench_data(original_file, "/mnt/sharefs/users/jianshu.she/ood__ifbench_95.1k_fixed.parquet")

if __name__ == "__main__":
    main() 