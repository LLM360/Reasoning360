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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict


class DAPORewardManager:
    """The reward manager.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 compute_score=None,
                 reward_fn_key='data_source',
                 max_resp_len=None,
                 overlong_buffer_cfg=None,
                 **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        original_compute_score = compute_score or _default_compute_score
        
        def debug_compute_score(*args, **kwargs):
            print(f"[DEBUG] compute_score called with data_source={kwargs.get('data_source')}")
            result = original_compute_score(*args, **kwargs)
            print(f"[DEBUG] compute_score returned {result}")
            return result
        
        self.compute_score = debug_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        print(f"[DEBUG] data.batch['responses'] shape: {data.batch['responses'].shape}")
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        print(f"[DEBUG] reward_tensor initial shape: {reward_tensor.shape}")

        # Add this to understand DataProto structure
        print(f"[DEBUG] DataProto length: {len(data)}")
        print(f"[DEBUG] DataProto batch keys: {list(data.batch.keys())}")
        print(f"[DEBUG] DataProto non_tensor_batch keys: {list(data.non_tensor_batch.keys())}")

        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        print(f"[DEBUG] Processing {len(data)} items")

        # Count data sources
        data_source_counts = defaultdict(int)
        for i in range(len(data)):
            data_source = data[i].non_tensor_batch[self.reward_fn_key]
            data_source_counts[data_source] += 1
        
        print(f"[DEBUG] Data source distribution: {dict(data_source_counts)}")

        # Check if any data is being filtered
        for i in range(len(data)):
            data_item = data[i]
            if self.reward_fn_key not in data_item.non_tensor_batch:
                print(f"[DEBUG] Warning: Item {i} missing reward_fn_key '{self.reward_fn_key}'")
            
            # Check if ground truth exists
            if 'reward_model' not in data_item.non_tensor_batch or 'ground_truth' not in data_item.non_tensor_batch['reward_model']:
                print(f"[DEBUG] Warning: Item {i} missing ground_truth")

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()  # qqq
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            print(f"[DEBUG] Item {i}, data_source: {data_source}")

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            print(f"[DEBUG] Computing score for data_source: {data_source}")
            try:
                result = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info
                )
                print(f"[DEBUG] Score computation successful: {result}")
            except Exception as e:
                print(f"[DEBUG] Error computing score for data_source {data_source}: {e}")
                raise

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    print(f"[DEBUG] in reward_extra_info, key: {key}, value: {value}")
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len  # 512
                expected_len = self.max_resp_len - overlong_buffer_len  # 16k-512=15488
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)  # qqq: there's no lower bound of `overlong_reward` as suggested in the paper
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            print(f"[DEBUG] Item {i}, valid_response_length: {valid_response_length}, reward: {reward}")
            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth) 
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        # print(f"[DEBUG] Final reward_tensor shape: {reward_tensor.shape}")
        # print(f"[DEBUG] Non-zero elements in reward_tensor: {(reward_tensor != 0).sum().item()}")
        # print(f"[DEBUG] Unique data sources processed: {list(already_print_data_sources.keys())}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor