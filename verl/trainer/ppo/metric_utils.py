# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List, Callable
import numpy as np
from verl import DataProto
from collections import Counter, defaultdict
from functools import partial
import wandb

# Add at module level (top of file with other imports)
_scores_tables = {}  # Global dictionary to store wandb tables

def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Group response lengths and rewards by data source
    data_source_response_lengths = defaultdict(list)
    data_source_scores = defaultdict(list)
    for i, data_source in enumerate(batch.non_tensor_batch['data_source']):
        data_source_response_lengths[data_source].append(response_length[i].item())
        data_source_scores[data_source].append(sequence_score[i].item())

    metrics = {
        # score
        "critic/score/mean":
            torch.mean(sequence_score).detach().item(),
        "critic/score/max":
            torch.max(sequence_score).detach().item(),
        "critic/score/min":
            torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean":
            torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max":
            torch.max(sequence_reward).detach().item(),
        "critic/rewards/min":
            torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean":
            torch.mean(valid_adv).detach().item(),
        "critic/advantages/max":
            torch.max(valid_adv).detach().item(),
        "critic/advantages/min":
            torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean":
            torch.mean(valid_returns).detach().item(),
        "critic/returns/max":
            torch.max(valid_returns).detach().item(),
        "critic/returns/min":
            torch.min(valid_returns).detach().item(),
        **({
            # values
            "critic/values/mean": torch.mean(valid_values).detach().item(),
            "critic/values/max": torch.max(valid_values).detach().item(),
            "critic/values/min": torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        "response_length/mean":
            torch.mean(response_length).detach().item(),
        "response_length/max":
            torch.max(response_length).detach().item(),
        "response_length/min":
            torch.min(response_length).detach().item(),
        "response_length/clip_ratio":
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean":
            torch.mean(prompt_length).detach().item(),
        "prompt_length/max":
            torch.max(prompt_length).detach().item(),
        "prompt_length/min":
            torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio":
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # Add data source specific response length metrics
    for data_source, lengths in data_source_response_lengths.items():
        lengths_tensor = torch.tensor(lengths)
        metrics.update({
            f"response_length/{data_source}/mean": torch.mean(lengths_tensor).item(),
            f"response_length/{data_source}/max": torch.max(lengths_tensor).item(),
            f"response_length/{data_source}/min": torch.min(lengths_tensor).item(),
            f"response_length/{data_source}/clip_ratio": torch.mean(torch.eq(lengths_tensor, max_response_length).float()).item(),
        })

    # Add data source specific reward metrics
    for data_source, scores in data_source_scores.items():
        scores_tensor = torch.tensor(scores)
        metrics.update({
            f"critic/scores/{data_source}/mean": torch.mean(scores_tensor).item(),
            f"critic/scores/{data_source}/max": torch.max(scores_tensor).item(),
            f"critic/scores/{data_source}/min": torch.min(scores_tensor).item(),
            f"critic/scores/{data_source}/std": torch.std(scores_tensor).item(),
        })

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{
            name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]
        },
    }

    return {
        **{
            f"timing_s/{name}": value for name, value in timing_raw.items()
        },
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }

def bootstrap_metric(data: list[dict[str, Any]],
                     subset_size: int,
                     reduce_fns: list[Callable[[np.ndarray], float]],
                     n_bootstrap: int = 1000,
                     seed: int = 42) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val

def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], 
                               infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.
    
    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infors_dict: variable name -> list of values for each sample
        
    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                        data=var_vals, 
                        subset_size=n, 
                        reduce_fns=[np.max, np.min], 
                        seed=seed)
                    
                    metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                    metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std

                    # Majority voting
                    if var2vals.get("pred", None) is not None:
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                        [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                            data=vote_data,
                            subset_size=n,
                            reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                            seed=seed,
                        )
                        metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    
    return data_src2var2metric2val

def compute_difficulty_histogram_metrics(batch: DataProto, config) -> Dict[str, Any]:
    metrics = {}
    
    with torch.no_grad():
        num_rollout = config.actor_rollout_ref.rollout.n
        sequence_score = batch.batch['token_level_scores'].sum(-1)  # batch_size
        uids = batch.non_tensor_batch['uid']
        sorted_indices = sorted(range(len(uids)), key=lambda i: uids[i])
        sorted_indices_tensor = torch.tensor(sorted_indices, device=sequence_score.device)
        sequence_score = sequence_score[sorted_indices_tensor]

        batch_score = sequence_score.reshape([-1, num_rollout]) # batch_size, num_rollout
        
        avg_batch_score_per_batch = torch.mean(batch_score, dim=-1) # batch_size
        avg_batch_score_per_batch_np = avg_batch_score_per_batch.detach().cpu().numpy().reshape([-1])

        # group the score by batch.non_tensor_batch['data_source']
        data_source_score_dict = defaultdict(list)
        
        # make batch_source_list sorted by uid, consistent with batch_score
        batch_source_list = batch.non_tensor_batch['data_source']
        sorted_batch_source_list = [batch_source_list[i] for i in sorted_indices]
        # bs * rollout -> bs
        sorted_batch_source_list = sorted_batch_source_list[::num_rollout]
        
        # both are sorted by uid
        for score, data_source in zip(avg_batch_score_per_batch_np, sorted_batch_source_list):
            data_source_score_dict[data_source].append(score)

        # add wandb histogram for each data source
        for data_source, scores in data_source_score_dict.items():
            metrics[f'acc_inter_val_per_batch/{data_source}/histogram'] = wandb.Histogram(sequence=scores, num_bins=10)
            
            # Create or get existing table
            table_key = f'scores_table_{data_source}'
            if table_key not in _scores_tables:
                _scores_tables[table_key] = wandb.Table(columns=["step", "score"])
            
            # Create new table with existing data
            new_table = wandb.Table(columns=["step", "score"], data=_scores_tables[table_key].data)
            
            # Add new scores
            for score in scores:
                new_table.add_data(batch.meta_info.get("step", 0), score)
                
            metrics[f'acc_inter_val_per_batch/{data_source}/scores'] = new_table
            _scores_tables[table_key] = new_table  # Update reference for next time

    # Overall histogram
    metrics['acc_inter_val_per_batch/histogram'] = wandb.Histogram(sequence=avg_batch_score_per_batch_np, num_bins=10)
    
    # Overall scores table
    if 'all_scores_table' not in _scores_tables:
        _scores_tables['all_scores_table'] = wandb.Table(columns=["step", "score"])
    
    # Create new table with existing data
    new_all_table = wandb.Table(columns=["step", "score"], data=_scores_tables['all_scores_table'].data)
    
    # Add new scores
    all_scores = avg_batch_score_per_batch_np.tolist()
    for score in all_scores:
        new_all_table.add_data(batch.meta_info.get("step", 0), score)
        
    metrics['acc_inter_val_per_batch/all_scores'] = new_all_table
    _scores_tables['all_scores_table'] = new_all_table  # Update reference for next time
    
    return metrics
