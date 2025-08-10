from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

# --- Helper Function for Parallel Computation (with Per-Job Timeout) ---


def parallel_compute_score(compute_score_fn, data_sources, solutions, ground_truths, extra_infos, num_processes=64, timeout_per_job=180):
    """
    Computes scores in parallel with a timeout for each individual job.

    Args:
        compute_score_fn: The function to compute the score for a single item.
        ... (other args)
        num_processes: The number of worker processes.
        timeout_per_job: A timeout in seconds for each individual call to
                         compute_score_fn. If a single job takes longer than this,
                         it will be aborted and its result will be None.

    Returns:
        A list of results, with None for timed-out or failed jobs.
    """
    results = [None] * len(solutions)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 1. Submit all jobs to the pool. executor.submit returns a Future object.
        #    A Future represents a computation that may or may not have completed.
        futures = {executor.submit(compute_score_fn, data_source=ds, solution_str=sol, ground_truth=gt, extra_info=ei): i for i, (ds, sol, gt, ei) in enumerate(zip(data_sources, solutions, ground_truths, extra_infos))}

        # 2. Retrieve results as they complete, with a per-job timeout.
        for future in futures:
            original_index = futures[future]
            try:
                # future.result() waits for the single job to finish.
                # The timeout here applies ONLY to this one job.
                result = future.result(timeout=timeout_per_job)
                results[original_index] = result
            except TimeoutError:
                print(f"Job {original_index} timed out after {timeout_per_job} seconds. Result is None.")
                # The result for this index remains None
            except Exception as e:
                print(f"Job {original_index} failed with an error: {e}. Result is None.")
                # The result for this index remains None

    return results


# --- Synchronous Reward Manager (Updated to use Per-Job Timeout) ---


class MultiProcessRewardManager:
    """
    The reward manager, now robust to individual process timeouts.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        shuffle_data=True,
        num_processes=64,
        timeout_per_job=180,  # Changed from timeout_per_batch
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.shuffle_data = shuffle_data
        self.num_processes = num_processes
        self.timeout_per_job = timeout_per_job  # Timeout for each individual job

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Computes rewards for a batch of data synchronously."""

        if "rm_scores" in data.batch.keys():
            return {"reward_tensor": data.batch["rm_scores"]} if return_dict else data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # --- Data Preparation (largely unchanged) ---
        # NOTE: Shuffling is now handled slightly differently because results
        # from executor.submit do not arrive in order. We map results
        # back to their original positions.
        data_sources, solutions, ground_truths, extra_infos = [], [], [], []
        valid_response_lengths, prompt_strs, response_strs = [], [], []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_lengths.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            prompt_strs.append(prompt_str)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            if response_str.endswith(self.tokenizer.eos_token):
                response_str = response_str[: -len(self.tokenizer.eos_token)]
            response_strs.append(response_str)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            data_sources.append(data_source)
            solutions.append(response_str)
            ground_truths.append(ground_truth)
            extra_infos.append(extra_info)

        # --- Run Synchronous Parallel Computation ---
        # The new function is called here with the per-job timeout.
        results = parallel_compute_score(
            self.compute_score,
            data_sources,
            solutions,
            ground_truths,
            extra_infos,
            num_processes=self.num_processes,
            timeout_per_job=self.timeout_per_job,  # Pass the per-job timeout
        )

        # --- Process Results ---
        # No need to handle shuffling here anymore, as results are mapped back to their original index.
        already_print_data_sources = {}
        for i in range(len(data)):
            result = results[i]  # Results are now in the correct original order

            # Get other data using the original index
            data_source = data_sources[i]
            response_str = response_strs[i]
            prompt_str = prompt_strs[i]
            ground_truth = ground_truths[i]
            valid_response_length = valid_response_lengths[i]

            score = 0.0
            if result is None:
                # This handles individual failures and timeouts
                result = {"score": 0.0, "acc": 0.0, "reason": "timeout_or_error"}
            elif not isinstance(result, dict):
                result = {"score": result, "acc": result}

            score = result.get("score", 0.0)
            for key, value in result.items():
                reward_extra_info[key].append(value)

            reward = score
            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                # ... (overlong penalty logic is unchanged) ...
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # Assign reward to the correct position in the final tensor
            reward_tensor[i, valid_response_length - 1] = reward

            if already_print_data_sources.get(data_source, 0) < self.num_examine:
                already_print_data_sources[data_source] = already_print_data_sources.get(data_source, 0) + 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for key, value in result.items():
                    print(f"[{key}]", value)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
