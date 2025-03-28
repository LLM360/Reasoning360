#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model uploader for Hugging Face Hub.

This script uploads model files to the Hugging Face Hub repository.
"""

import os
import argparse
import logging
from typing import Optional
from huggingface_hub import HfApi, login


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Name of the repository on Hugging Face Hub",
    )
    parser.add_argument(
        "--is_private",
        action="store_true",
        help="Whether the repository should be private",
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_or_get_repo(api: HfApi, repo_id: str, is_private: bool) -> None:
    """Create a repository if it doesn't exist or get an existing one.

    Args:
        api: Hugging Face API instance.
        repo_id: Repository identifier.
        is_private: Whether the repository should be private.
    """
    api.create_repo(repo_id=repo_id, exist_ok=True, private=is_private)
    logging.info(f"Repository {repo_id} is ready for uploads")


def upload_model_files(api: HfApi, model_dir: str, repo_id: str) -> None:
    """Upload all files from the model directory to the repository.

    Args:
        api: Hugging Face API instance.
        model_dir: Path to the model directory.
        repo_id: Repository identifier.
    """
    for root, _, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, model_dir)
            logging.info(f"Uploading {relative_path}...")

            # continue

            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=relative_path,
                    repo_id=repo_id,
                    commit_message=f"Upload {relative_path}",
                )
                logging.info(f"Successfully uploaded {relative_path}")
            except Exception as e:
                logging.error(f"Failed to upload {relative_path}: {str(e)}")


def main() -> None:
    """Main function to execute the script."""
    setup_logging()
    args = parse_arguments()

    try:
        login()
        api = HfApi()

        create_or_get_repo(api, args.repo_name, args.is_private)
        upload_model_files(api, args.model_path, args.repo_name)

        logging.info("Model upload completed successfully")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
