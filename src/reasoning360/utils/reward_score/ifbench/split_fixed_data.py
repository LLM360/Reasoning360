#!/usr/bin/env python3
"""
Split the fixed IFBench data into train and test sets.
"""

import pandas as pd
import os

def split_fixed_data():
    """Split the fixed IFBench data into 90% train and 10% test."""
    
    # Input file (fixed data)
    input_file = "/mnt/sharefs/users/jianshu.she/ood__ifbench_95.1k_fixed.parquet"
    
    # Output directory
    output_dir = "/mnt/sharefs/users/jianshu.she/ifbench_split"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    train_file = os.path.join(output_dir, "ifbench_train_fixed.parquet")
    test_file = os.path.join(output_dir, "ifbench_test_fixed.parquet")
    
    print(f"Reading fixed data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Total rows: {len(df)}")
    
    # Split data: 90% train, 10% test
    train_size = int(0.9 * len(df))
    test_size = len(df) - train_size
    
    print(f"Splitting into {train_size} train samples and {test_size} test samples")
    
    # Split the dataframe
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Save split data
    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)
    
    print(f"Train data saved to: {train_file}")
    print(f"Test data saved to: {test_file}")
    
    # Verify the split
    print(f"\nVerification:")
    print(f"Train file rows: {len(pd.read_parquet(train_file))}")
    print(f"Test file rows: {len(pd.read_parquet(test_file))}")
    
    # Check extra_info in split files
    train_data = pd.read_parquet(train_file)
    test_data = pd.read_parquet(test_file)
    
    print(f"\nTrain file - None values in extra_info: {train_data['extra_info'].isna().sum()}")
    print(f"Test file - None values in extra_info: {test_data['extra_info'].isna().sum()}")

if __name__ == "__main__":
    split_fixed_data() 