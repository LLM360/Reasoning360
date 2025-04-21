import argparse
import json
import os
import sys
import signal
import multiprocessing
from multiprocessing import Process
from functools import partial
import pickle

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port

from verl.utils.reward_score import _default_compute_score

class DifficultyFilterPipeline:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.sampling_params = None

    def initialize_components(self):
        """Initialize model, tokenizer and sampling parameters"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        
        self.model = LLM(
            self.args.model_path, 
            tensor_parallel_size=self.args.tp_size,
            enforce_eager=True,
        )
        
        self.sampling_params = SamplingParams(
            n=self.args.n,
            max_tokens=self.args.max_new_tokens,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            temperature=self.args.temperature,
            repetition_penalty=self.args.repetition_penalty,
            skip_special_tokens=True,
        )

    def prepare_dataset(self):
        """Load and prepare dataset for inference"""
        # Load parquet file directly using datasets library
        dataset = Dataset.from_parquet(self.args.dataset_parquet_path)
        
        if self.args.debug:
            dataset = dataset.select(range(min(48, len(dataset))))
            print(f"[DEBUG] Using first {min(48, len(dataset))} samples for testing")
        
        # Handle data parallelism
        if self.args.dp_size > 1:
            total = len(dataset)
            per_rank = total // self.args.dp_size
            start = self.args.dp_rank * per_rank
            end = start + per_rank if self.args.dp_rank != self.args.dp_size - 1 else total
            dataset = dataset.select(range(start, end))
            print(f"DP rank {self.args.dp_rank} processing {len(dataset)} samples out of {total}")
        
        return dataset

    def get_checkpoint_path(self):
        """Get the path for the checkpoint file"""
        rank_output_dir = os.path.join(self.args.output_dir, f"dp{self.args.dp_rank}")
        return os.path.join(rank_output_dir, "checkpoint.pkl")

    def save_checkpoint(self, current_batch_idx, global_results, global_errors):
        """Save checkpoint for resuming later"""
        checkpoint_path = self.get_checkpoint_path()
        checkpoint_data = {
            "current_batch_idx": current_batch_idx,
            "global_results": global_results,
            "global_errors": global_errors
        }
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        print(f"Checkpoint saved at batch {current_batch_idx} to {checkpoint_path}")

    def load_checkpoint(self):
        """Load checkpoint if it exists"""
        checkpoint_path = self.get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "rb") as f:
                    checkpoint_data = pickle.load(f)
                print(f"Resuming from checkpoint at batch {checkpoint_data['current_batch_idx']}")
                return checkpoint_data
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from the beginning.")
        return {"current_batch_idx": 0, "global_results": {}, "global_errors": {}}

    def run_inference(self):
        """Main pipeline execution"""
        self.initialize_components()
        dataset = self.prepare_dataset()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=min(4, os.cpu_count() // self.args.dp_size),
            prefetch_factor=8
        )
        
        # Load checkpoint if it exists
        checkpoint_data = self.load_checkpoint()
        start_batch_idx = checkpoint_data["current_batch_idx"]
        global_results = checkpoint_data["global_results"]
        global_errors = checkpoint_data["global_errors"]
        
        rank_output_dir = os.path.join(self.args.output_dir, f"dp{self.args.dp_rank}")
        os.makedirs(rank_output_dir, exist_ok=True)

        # Setup progress tracking
        total_batches = len(dataloader)
        progress_bar = tqdm(total=total_batches, initial=start_batch_idx, desc=f"DP{self.args.dp_rank} Processing")

        # Skip already processed batches
        batch_iterator = iter(dataloader)
        for _ in range(start_batch_idx):
            try:
                next(batch_iterator)
            except StopIteration:
                print(f"Warning: Checkpoint index {start_batch_idx} exceeds dataset size {total_batches}")
                break

        for idx, batch_dict in enumerate(batch_iterator, start=start_batch_idx):
            if self.args.debug:
                print(f"Processing batch {idx}...")
                print(batch_dict)
            
            batch_size = len(batch_dict["reward_model"])
            model_inputs = []
            for i, apply_template in enumerate(batch_dict.get("apply_chat_template", [False] * batch_size)):
                if apply_template:
                    # Use the chat template prompt if specified
                    prompt = batch_dict["prompt"][i]
                    model_inputs.append({"prompt": self.tokenizer.apply_chat_template(prompt, tokenize=False)})
                else:
                    # Use the raw prompt
                    raw_prompt = batch_dict["raw_prompt"][i]
                    model_inputs.append({"prompt": raw_prompt})

            keys = batch_dict.keys()
            
            print(f"Generating for batch {idx}...")
            outputs = self.model.generate(
                model_inputs, 
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            print(f"Done Generating for batch {idx}!")

            batch_results = self.process_batch_outputs(outputs, batch_dict)

            for i, result in batch_results.items():
                global_results[f"{idx}_{i}"] = result
            
            batch_output_path = os.path.join(rank_output_dir, f"batch_{idx:05d}.json")
            with open(batch_output_path, "w") as f:
                json.dump(batch_results, f, indent=2)

            # Save error if it exists
            if idx in global_errors:
                batch_error_path = os.path.join(rank_output_dir, f"batch_{idx:05d}_error.json")
                with open(batch_error_path, "w") as f:
                    json.dump({idx: global_errors[idx]}, f, indent=2)
            
            # Save checkpoint every few batches for auto-resuming
            if idx % self.args.checkpoint_freq == 0:
                self.save_checkpoint(idx + 1, global_results, global_errors)
                
            progress_bar.update(1)


        progress_bar.close()
        
        # Save final checkpoint
        self.save_checkpoint(total_batches, global_results, global_errors)
        
        # Save consolidated results
        final_results_path = os.path.join(rank_output_dir, "final_results.json")
        with open(final_results_path, "w") as f:
            json.dump({
                "results": global_results,
                "errors": global_errors
            }, f, indent=2)
        
        print(f"Completed processing {total_batches} batches. Final results saved to {final_results_path}")
        return global_results

    def process_batch_outputs(self, outputs, batch_dict):
        """Process model outputs for a batch using simple reward computation"""
        batch_results = {}
        
        for i in range(len(outputs)):
            responses = [response.text for response in outputs[i].outputs]
            
            data_source = batch_dict["data_source"][i]
            ground_truth = batch_dict["reward_model"]["ground_truth"][i]
            question = batch_dict["raw_prompt"][i]
            
            try:
                # Compute scores directly using _default_compute_score
                scores = []
                detailed_scores = []
                
                for response in responses:
                    
                    score = _default_compute_score(
                        data_source=data_source,
                        solution_str=response,
                        ground_truth=ground_truth,
                        extra_info={key: batch_dict["extra_info"][key][i] for key in batch_dict["extra_info"]}
                    )
                    scores.append(score)
                    detailed_scores.append({"score": score})
                
                # Calculate how many responses meet the threshold
                pass_count = sum(1 for score in scores if score >= self.args.correct_reward_threshold)
                
            except Exception as e:
                print(f"Error computing scores: {e}")
                scores = [0.0] * len(responses)
                detailed_scores = [{"score": 0.0, "error": str(e)}] * len(responses)
                pass_count = 0
            
            # Format the batch results
            batch_results[i] = {
                "question": question,
                "ground_truth": ground_truth,
                "source": data_source,
                "responses": responses,
                "scores": scores,
                "detailed_scores": detailed_scores,
                "pass_rate": pass_count / len(responses) if responses else 0,
            }
            
        return batch_results

    @staticmethod
    def _log_error(log_file, message):
        with open(log_file, "a") as f:
            f.write(f"{message}\n")

def run_dp_worker(args, dp_rank, dp_size):
    """Run data parallel worker process"""
    # Set CUDA devices for this worker
    gpu_offset = dp_rank * args.tp_size
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_offset + i) for i in range(args.tp_size))
    torch.cuda.set_device(0)  # Use first visible device

    # Set vLLM environment variables
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = args.master_addr
    os.environ["VLLM_DP_MASTER_PORT"] = str(args.master_port)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_TORCH_DIST"] = "1"
    
    print(f"[DP rank {dp_rank}] sees {torch.cuda.device_count()} visible GPU(s).")

    # Update args with DP rank info
    args.dp_rank = dp_rank
    args.dp_size = dp_size
    
    # Run inference pipeline
    pipeline = DifficultyFilterPipeline(args)
    pipeline.run_inference()

def main():
    """Main entry point"""
    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        sys.exit(1)
        
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    parser = argparse.ArgumentParser(description="Difficulty filtering using verl reward functions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_parquet_path", type=str, required=True, help="Path to the dataset parquet file")
    parser.add_argument("--output_dir", type=str, default="./diff_filter_output", help="Output directory")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Key for prompt in the dataset")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Maximum prompt length")
    parser.add_argument("--truncation", type=str, default="error", choices=["left", "right", "error"], 
                       help="How to handle prompts exceeding max_prompt_length")
    parser.add_argument("--default_data_source", type=str, default="None", 
                       help="Default data source for reward function")
    parser.add_argument("--correct_reward_threshold", type=float, default=1.0, 
                       help="Threshold for considering an answer correct")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--n", type=int, default=16, help="Number of generations per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--top_k", type=int, default=100, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")  # 1.0 means no repetition penalty
    parser.add_argument("--checkpoint_freq", type=int, default=5, help="Frequency to save checkpoints (in batches)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (uses small dataset subset)")
    
    # Data parallelism / multi-node arguments
    parser.add_argument("--dp_size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--node_size", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of the current node")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Master node IP address")
    parser.add_argument("--master_port", type=int, default=0, help="Master node port")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Set multiprocessing start method
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Set up DP master information
    if args.node_size == 1:
        args.master_port = get_open_port()
    else:
        assert args.master_addr != "127.0.0.1", "For multi-node, --master_addr must be provided."
        assert args.master_port > 0, "For multi-node, --master_port must be provided."

    dp_size = args.dp_size
    node_size = args.node_size
    node_rank = args.node_rank
    
    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    if dp_size == 1:
        # Single worker process
        run_dp_worker(args, dp_rank=0, dp_size=1)
    else:
        # Multiple worker processes
        procs = []
        for local_dp_rank in range(dp_per_node):
            global_dp_rank = node_rank * dp_per_node + local_dp_rank
            proc = Process(target=run_dp_worker, args=(args, global_dp_rank, dp_size))
            proc.start()
            procs.append(proc)
        
        exit_code = 0
        for proc in procs:
            proc.join()
            if proc.exitcode is None:
                print(f"Killing process {proc.pid} that didn't stop.")
                proc.kill()
                exit_code = 1
            elif proc.exitcode:
                exit_code = proc.exitcode
        
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
    
    # example:
    # python diff_filter.py --model_path "/lustrefs/users/shibo.hao/hf_models/Qwen2.5-3B-Instruct" --dataset_parquet_path "/lustrefs/users/shibo.hao/Reasoning360/data/train/math__bigmath_filtered_mar21_10k.parquet" --output_dir "./diff_filter_output" --prompt_key "prompt" --max_prompt_length 2048 --truncation "left" --checkpoint_freq 5 --debug --dp_size 1 --tp_size 1 --node_size 1 --node_rank 0 --master_addr "127.0.0.1" --master_port 0