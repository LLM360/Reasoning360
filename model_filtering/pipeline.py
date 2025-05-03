#!/usr/bin/env python3

# Standard library imports
import glob
import json
import os
import time
import datetime
from datetime import timedelta

# Third party imports
import numpy as np
from datasets import Dataset
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import traceback
# Local imports
from model_filtering.utils import console, json_default, custom_collate_fn

class DifficultyFilterPipeline:
    def __init__(self, args, dataset=None):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.sampling_params = None

        self.start_time = time.time()
        self.gen_times = []
        self.dataset = dataset

    @staticmethod
    def format_time(seconds):
        return str(timedelta(seconds=int(seconds)))

    # ------------- component init ------------------------------------------ #
    def initialize_components(self):
        console.print(f"🔄 Loading tokenizer from [highlight]{self.args.model_path}[/highlight]...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

        console.print(
            f"🚀 Initializing model from [highlight]{self.args.model_path}[/highlight] "
            f"with TP={self.args.tp_size}..."
        )
        console.print(f"Warning: enable_expert_parallel is set to {self.args.enable_expert_parallel}")
        console.print("If you are using MOE models, set enable_expert_parallel to True")
        
        self.model = LLM(
            self.args.model_path,
            tensor_parallel_size=self.args.tp_size,
            enable_expert_parallel=self.args.enable_expert_parallel,
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
            truncate_prompt_tokens=self.args.max_prompt_length
        )
        console.print("✅ Model initialization [success]complete[/success].")

    # ------------- dataset -------------------------------------------------- #
    def prepare_dataset(self):
        console.print(f"📂 Loading dataset from [highlight]{self.args.dataset_parquet_path}[/highlight]...")
        console.print(f"🐞 [DEBUG] Dataset loaded with [highlight]{len(self.dataset)}[/highlight] samples")

        assert self.dataset is None

        # TODO: replace None values with empty strings in dataset columns and nested dictionaries
        # TODO: Below sometimes causes stucks in the process
        # # Replace None values with empty strings in dataset columns and nested dictionaries
        # def replace_none_with_empty(obj, max_depth=3):
        #     if obj is None:
        #         return ""
        #     elif isinstance(obj, dict) and max_depth > 0:
        #         return {k: replace_none_with_empty(v, max_depth-1) for k, v in obj.items()}
        #     elif isinstance(obj, list) and max_depth > 0:
        #         return [replace_none_with_empty(item, max_depth-1) for item in obj]
        #     return obj

        # def process_example(example):
        #     return {k: replace_none_with_empty(v) for k, v in example.items()}

        # # Use batched processing with multiple workers
        # dataset = dataset.map(
        #     process_example,
        #     num_proc=min(8, os.cpu_count()),
        #     desc="Processing None values"
        # )
        # console.print("🧹 Replaced None values with empty strings using parallel processing")
            
        # ── debug slice
        if self.args.debug:
            self.dataset = self.dataset.select(range(min(48, len(self.dataset))))
            console.print(f"🐞 [DEBUG] Using first [highlight]{len(self.dataset)}[/highlight] samples")

        # ── KEEP-ONLY columns actually referenced downstream ---------------- #
        required_cols = {
            "prompt",
            "reward_model",
            "apply_chat_template",
            "data_source",
            "extra_info",
        }
        cols_to_drop = [c for c in self.dataset.column_names if c not in required_cols]
        if cols_to_drop:
            self.dataset = self.dataset.remove_columns(cols_to_drop)
            console.print(f"🧹 Dropped {len(cols_to_drop)} column(s) for easier processing: {', '.join(cols_to_drop)}")


    # ------------- checkpoint paths / I-O ------------------------------ #
    def get_checkpoint_path(self):
        model_name = self.args.model_path.split("/")[-1]
        dataset_name = os.path.basename(self.args.dataset_parquet_path).rsplit(".parquet", 1)[0]
        rank_output_dir = os.path.join(self.args.output_dir, dataset_name, model_name, f"dp{self.args.dp_rank}")
        os.makedirs(rank_output_dir, exist_ok=True)
        return os.path.join(rank_output_dir, "checkpoint.pkl")

    def load_checkpoint(self):
        model_name = self.args.model_path.split("/")[-1]
        dataset_name = os.path.basename(self.args.dataset_parquet_path).rsplit(".parquet", 1)[0]
        rank_output_dir = os.path.join(self.args.output_dir, dataset_name, model_name, f"dp{self.args.dp_rank}")
        os.makedirs(rank_output_dir, exist_ok=True)
        
        # If force_regenerate is set, start from scratch
        if self.args.force_regenerate:
            console.print("🔄 [warning]Force regeneration requested - ignoring existing checkpoints[/warning]")
            return {"current_batch_idx": 0, "global_results": {}, "global_errors": {}}
        
        # Find batch files and extract their numbers
        batch_files = glob.glob(os.path.join(rank_output_dir, "batch_*.json"))
        batch_files_with_idx = []
        
        for batch_file in batch_files:
            try:
                batch_num = int(os.path.basename(batch_file).split("_")[1].split(".")[0])
                batch_files_with_idx.append((batch_num, batch_file))
            except (ValueError, IndexError):
                console.print(f"⚠️ [warning]Could not parse batch number from {batch_file}[/warning]")
        
        # Sort by batch number
        batch_files_with_idx.sort(key=lambda x: x[0])
        
        # Load results from existing batch files
        global_results = {}
        global_errors = {}
        
        if batch_files_with_idx:
            console.print(f"📋 Found {len(batch_files_with_idx)} existing batch files to load")
            
            for batch_idx, batch_file in batch_files_with_idx:
                try:
                    with open(batch_file, 'r') as f:
                        batch_data = json.load(f)
                    # Directly load results (no reward info expected here)
                    for i, result in batch_data.items():
                        global_results[f"{batch_idx}_{i}"] = result
                    console.print(f"✅ Loaded batch {batch_idx} with {len(batch_data)} results")
                except Exception as e:
                    console.print(f"⚠️ [warning]Failed to load batch file {batch_file}: {e}[/warning]")
        
        # Determine the next batch index to start from
        start_batch_idx = batch_files_with_idx[-1][0] + 1 if batch_files_with_idx else 0
        
        if batch_files_with_idx:
            console.print(f"📋 Resuming from batch {start_batch_idx} with {len(global_results)} loaded results")
        
        return {"current_batch_idx": start_batch_idx, "global_results": global_results, "global_errors": global_errors}

    # ------------- batch serialization  ------------------------------------ #
    def process_batch_outputs(self, outputs, batch_dict):
        batch_results = {}
        console.print(f"DP{self.args.dp_rank} Begin to process {len(outputs)} samples")

        tokenizer = self.model.get_tokenizer()
        MAX_LEN = self.args.max_prompt_length

        for i in range(len(outputs)):
            messages = batch_dict["prompt"][i]
            text = "".join(m["content"] for m in messages)
            token_len = len(tokenizer(text).input_ids)

            # build extra_info, adding flags and lengths
            extra_info = {k: batch_dict["extra_info"][k][i]
                        for k in batch_dict["extra_info"]}
            extra_info["too_long"] = (token_len > MAX_LEN)
            extra_info["token_length"] = token_len

            try:
                # collect model outputs
                full_responses = [r.text for r in outputs[i].outputs]

                # compute mean response length in tokens
                if full_responses:
                    resp_token_counts = [
                        len(tokenizer(resp).input_ids) for resp in full_responses
                    ]
                    mean_resp_len = sum(resp_token_counts) / len(resp_token_counts)
                else:
                    mean_resp_len = 0.0
                extra_info["response_mean_length"] = mean_resp_len

                data_source  = batch_dict["data_source"][i]
                if "zebra" in self.args.dataset_parquet_path:
                    ground_truth = ""
                else:
                    ground_truth = batch_dict["reward_model"]["ground_truth"][i]
                question     = next((m["content"] for m in messages if m["role"] == "user"),
                                    "No question found")

                batch_results[i] = {
                    "messages":     messages,
                    "question":     question,
                    "ground_truth": ground_truth,
                    "source":       data_source,
                    "responses":    full_responses,
                    "extra_info":   extra_info,
                }

                if token_len > MAX_LEN:
                    console.print(f"⚠️ Sample {i} is over-length ({token_len} tokens); marked in extra_info.")

            except Exception as e:
                console.print(f"❌ Error processing sample {i}: {e}")
                console.print(traceback.format_exc())

        console.print(f"✅ Finished processing. Kept {len(batch_results)} / {len(outputs)} samples.")
        return batch_results

    # ------------- progress ------------------------------------------------- #
    def print_progress_stats(self, idx, total_batches):
        elapsed = time.time() - self.start_time
        eta = "calculating..."
        if idx > 0 and self.gen_times:
            remain = total_batches - idx - 1
            eta_batch = np.mean(self.gen_times[-10:])
            eta = self.format_time(remain * eta_batch)

        console.print()
        console.print(
            Panel(
                f"[bold]Progress: [metric]{idx+1}/{total_batches}[/metric] "
                f"({(idx+1)/total_batches*100:.1f}%)[/bold]",
                title="📊 Summary",
                border_style="cyan",
            )
        )
        console.print(f"⏱️  Elapsed: [time]{self.format_time(elapsed)}[/time] | ETA: [time]{eta}[/time]")
        if self.gen_times:
            console.print(f"⚡ Generation avg (last 10): [metric]{np.mean(self.gen_times[-10:]):.2f}s[/metric]")

    def wait_for_all_ranks(self, base_output_dir):
        """
        Wait for all DP ranks to complete their processing.
        
        Args:
            base_output_dir: Base directory for all DP ranks output
        """
        if self.args.dp_size <= 1:
            return
            
        console.print(f"\n⏳ [DP-{self.args.dp_rank}] Waiting for all DP ranks to complete...\n")
        
        # Each rank checks for completion files from all other ranks
        all_ranks_ready = False
        sleep_time = 5  # Start with 5 seconds
        max_sleep_time = 60  # Maximum sleep time of 60 seconds
        check_count = 0
        
        while not all_ranks_ready:
            # Check if all ranks have completed
            completed_ranks = []
            missing_ranks = []
            
            for rank in range(self.args.dp_size):
                rank_dir = os.path.join(base_output_dir, f"dp{rank}")
                rank_completion_file = os.path.join(rank_dir, "RANK_COMPLETE")
                if os.path.exists(rank_completion_file):
                    completed_ranks.append(rank)
                else:
                    missing_ranks.append(rank)
            
            if len(completed_ranks) == self.args.dp_size:
                all_ranks_ready = True
                
                # Create a nice table summarizing all ranks
                table = Table(title=f"[DP-{self.args.dp_rank}] DP Ranks Completion Status")
                table.add_column("Rank", style="cyan")
                table.add_column("Status", style="green")
                
                for rank in range(self.args.dp_size):
                    table.add_row(f"DP-{rank}", "✅ Complete")
                
                console.print(table)
                console.print(f"\n✅ [DP-{self.args.dp_rank}] All {self.args.dp_size} DP ranks have completed their work")
                break
            
            # Create a table showing status of all ranks
            table = Table(title=f"[DP-{self.args.dp_rank}] DP Ranks Completion Status")
            table.add_column("Rank", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Wait Time", style="blue")
            
            for rank in range(self.args.dp_size):
                if rank in completed_ranks:
                    table.add_row(f"DP-{rank}", "✅ Complete", "-")
                else:
                    table.add_row(f"DP-{rank}", "⏳ Waiting", f"{sleep_time}s")
            
            console.print(table)
            console.print(f"\n⏳ [DP-{self.args.dp_rank}] Waiting for {len(missing_ranks)} rank(s) to complete... ({len(completed_ranks)}/{self.args.dp_size} done)")
            console.print(f"   [DP-{self.args.dp_rank}] Missing: {', '.join(f'DP-{r}' for r in missing_ranks)}")
            console.print(f"   [DP-{self.args.dp_rank}] Sleep time: {sleep_time}s (increases by 5s each check, max 60s)")
            
            # Sleep with adaptive time
            time.sleep(sleep_time)
            
            # Increase sleep time for next iteration
            check_count += 1
            sleep_time = min(5 + (check_count * 5), max_sleep_time)  # Add 5s each time, cap at max_sleep_time

    # ------------- main loop ------------------------------------------------ #
    def run_inference(self):
        self.initialize_components()
        self.prepare_dataset()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=min(4, os.cpu_count() // max(1, self.args.dp_size)),
            prefetch_factor=8,
            collate_fn=custom_collate_fn
        )

        chk = self.load_checkpoint()
        start_batch_idx = chk["current_batch_idx"]
        global_results  = chk["global_results"]
        global_errors   = chk["global_errors"]

        model_name   = self.args.model_path.split("/")[-1]
        dataset_name = os.path.basename(self.args.dataset_parquet_path).rsplit(".parquet", 1)[0]
        rank_output_dir = os.path.join(
            self.args.output_dir, dataset_name, model_name, f"dp{self.args.dp_rank}"
        )
        os.makedirs(rank_output_dir, exist_ok=True)

        progress_bar = tqdm(
            total=len(dataloader),
            initial=start_batch_idx,
            desc=f"🔄 DP{self.args.dp_rank} Processing",
            position=0,
        )

        batch_iter = iter(dataloader)
        for _ in range(start_batch_idx):
            try:
                next(batch_iter)
            except StopIteration:
                console.print(f"⚠️ [DP-{self.args.dp_rank}] [warning]Checkpoint index {start_batch_idx} exceeds dataset size[/warning]")
                break

        for idx, batch_dict in enumerate(batch_iter, start=start_batch_idx):
            batch_size = len(batch_dict["reward_model"]["ground_truth"])
            console.print(f"\n🔄 [DP-{self.args.dp_rank}] Generating for batch {idx}/{len(dataloader)-1} ({batch_size} samples)…")            
            
            # enforce apply_chat_template==True
            if not all(batch_dict.get("apply_chat_template", [True] * batch_size)):
                raise RuntimeError(
                    "Encountered apply_chat_template=False but raw_prompt column is removed. "
                    "Please ensure all samples set apply_chat_template=True."
                )

            gen_start = time.time()
            outputs = self.model.chat(
                    batch_dict["prompt"],
                    sampling_params=self.sampling_params
            )
            
            self.gen_times.append(time.time() - gen_start)
            console.print(f"⏱️  [DP-{self.args.dp_rank}] Generation took [time]{self.gen_times[-1]:.2f}s[/time]")

            # ----------- store outputs ------------------------- #
            batch_results = self.process_batch_outputs(outputs, batch_dict)
            global_results.update({f"{idx}_{i}": r for i, r in batch_results.items()})

            batch_out = os.path.join(rank_output_dir, f"batch_{idx:05d}.json")
            with open(batch_out, "w") as f:
                json.dump(batch_results, f, indent=2, default=json_default)
            console.print(f"💾 [DP-{self.args.dp_rank}] Saved batch results to [highlight]{batch_out}[/highlight]")

            self.print_progress_stats(idx, len(dataloader))
            progress_bar.update(1)
            console.rule(style="cyan")

        progress_bar.close()
        elapsed_total = time.time() - self.start_time

        console.print()
        console.print(Panel(f"[bold][DP-{self.args.dp_rank}] Inference completed!", title="🏁 Finished", border_style="green"))
        console.print(f"⏱️  [DP-{self.args.dp_rank}] Total time: [time]{self.format_time(elapsed_total)}[/time]")
        console.print(
            f"ℹ️ [DP-{self.args.dp_rank}] All per-batch JSON files are ready. "
            "Run the separate reward script to score and assemble final results."
        )
        
        # Create a completion marker for this rank
        model_name = self.args.model_path.split("/")[-1]
        dataset_name = os.path.basename(self.args.dataset_parquet_path).rsplit(".parquet", 1)[0]
        base_output_dir = os.path.join(self.args.output_dir, dataset_name, model_name)
        completion_file = os.path.join(rank_output_dir, "RANK_COMPLETE")
        with open(completion_file, 'w') as f:
            f.write(f"Completed at {datetime.datetime.now().isoformat()}")
        
        console.print(f"✅ [DP-{self.args.dp_rank}] Created completion marker for DP rank {self.args.dp_rank}")
        
        # Wait for all DP ranks to complete if needed
        if self.args.dp_size > 1:
            self.wait_for_all_ranks(base_output_dir)
            
        console.print(f"🏁 [DP-{self.args.dp_rank}] Process complete!")