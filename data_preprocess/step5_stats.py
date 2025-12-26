import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from pyarrow import parquet as pq

folder = "/lustrefs/users/haonan.li/Reasoning360/final"


def check_write_permission(folder_path: Path) -> bool:
    """Check if we have write permission to the folder."""
    try:
        # Try to create a temporary file to test write permission
        test_file = folder_path / ".write_test_temp"
        test_file.touch()
        test_file.unlink()  # Clean up the test file
        return True
    except (PermissionError, OSError):
        return False


def collect_row_counts_for_folder(folder_path: Path) -> List[Dict[str, int]]:
    """Collect row counts for each .parquet file directly under a folder."""
    results: List[Dict[str, int]] = []
    for file_path in sorted(folder_path.glob("*.parquet")):
        parquet_file = pq.ParquetFile(str(file_path))
        num_rows = parquet_file.metadata.num_rows
        results.append({"filename": file_path.name, "rows": num_rows})
    return results


def write_table_report(folder_path: Path, rows_info: List[Dict[str, int]], total_rows: int) -> Path:
    """Write a formatted table report in the folder and return its path."""
    if not rows_info:
        return folder_path / "data_summary.txt"
    
    # Create DataFrame with percentage column
    df = pd.DataFrame(rows_info).sort_values(by="filename")
    folder_total = df['rows'].sum()
    df['percentage'] = (df['rows'] / folder_total * 100).round(2)  # Percentage within this folder
    
    # Calculate column widths dynamically
    max_filename_len = max(len(row['filename']) for row in rows_info)
    filename_width = max(max_filename_len + 2, 50)  # At least 50 chars, add padding
    rows_width = 12
    percentage_width = 10
    
    total_width = filename_width + rows_width + percentage_width + 6  # 6 for spacing
    
    # Write as formatted text table
    output_path = folder_path / "data_summary.txt"
    with open(output_path, 'w') as f:
        f.write("=" * total_width + "\n")
        f.write(f"DATA SUMMARY FOR: {folder_path.name}\n")
        f.write("=" * total_width + "\n")
        f.write(f"{'Filename':<{filename_width}} {'Rows':<{rows_width}} {'Percentage':<{percentage_width}}\n")
        f.write("-" * total_width + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['filename']:<{filename_width}} {row['rows']:>{rows_width-2},} {row['percentage']:>{percentage_width-1}.2f}%\n")
        
        f.write("-" * total_width + "\n")
        f.write(f"{'TOTAL':<{filename_width}} {df['rows'].sum():>{rows_width-2},} {100.00:>{percentage_width-1}.2f}%\n")
        f.write("=" * total_width + "\n")
    
    return output_path


def main() -> None:
    # For each immediate subfolder under base_folder, create a table report with row counts
    base_folder = Path(folder)
    if not base_folder.exists():
        raise FileNotFoundError(f"Base folder not found: {base_folder}")

    # First pass: collect all data to calculate total
    all_data = {}
    total_rows = 0
    processed_folders = 0
    
    for entry in sorted(base_folder.iterdir()):
        if not entry.is_dir():
            continue
        
        # Check write permission before processing
        if not check_write_permission(entry):
            print(f"Skipping {entry} - no write permission")
            continue
            
        print(f"Processing {entry}")
        rows_info = collect_row_counts_for_folder(entry)
        
        # Store data for later processing
        all_data[entry] = rows_info
        folder_total = sum(row_info["rows"] for row_info in rows_info)
        total_rows += folder_total
        processed_folders += 1
        print(f"  -> {folder_total:,} rows in {len(rows_info)} files")
    
    # Second pass: write reports with percentage calculations
    print(f"\nWriting reports with percentages...")
    for entry, rows_info in all_data.items():
        table_path = write_table_report(entry, rows_info, total_rows)
        print(f"  -> Report written to {table_path}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Processed {processed_folders} folders")
    print(f"Total rows across all folders: {total_rows:,}")


if __name__ == "__main__":
    main()