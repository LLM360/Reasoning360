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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version

# Added by Reasoning360
import types

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][
        0
    ]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(
    value: Union[torch.Tensor, np.ndarray], repeats: int
) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(
        self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (
            not config.enforce_eager and config.free_cache_engine
        ), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert (
            tensor_parallel_size <= torch.distributed.get_world_size()
        ), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp", None) is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        assert (
            model_hf_config.max_position_embeddings
            >= config.prompt_length + config.response_length
        ), "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )
        # NOTE: added by Reasoning360
        self._monkey_patch_vllm_engine_v0()

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @staticmethod
    @contextmanager
    def timer():
        import time

        start = end = time.perf_counter()
        yield lambda: end - start
        end = time.perf_counter()

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3")
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ],
                dtype=object,
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
            ):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                    }
                )
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids}
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)

        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # NOTE: added by Reasoning360
        if "num_samples" in prompts.meta_info:
            kwargs["n"] = prompts.meta_info["num_samples"]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs), self.timer() as t:
            # NOTE: added by Reasoning360. partial rollout support
            enable_partial_rollout = False
            if getattr(self.config, "use_partial_rollout", False):
                partial_rollout_len = self.config.partial_rollout_len
                enable_partial_rollout = True
            # NOTE: partial rollout should not sample multiple responses
            if enable_partial_rollout:
                prompt_is_partial = prompts.batch['is_partial'].to('cpu')
                raw_sampling_params = self.sampling_params.clone()
                raw_sampling_params.max_tokens = partial_rollout_len
                partial_sampling_params = raw_sampling_params.clone()
                partial_sampling_params.n = 1
                sampling_params = [
                    partial_sampling_params if is_partial else raw_sampling_params
                    for is_partial in prompt_is_partial
                ]
            else:
                sampling_params = self.sampling_params

            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=sampling_params,
                use_tqdm=False)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            response_is_partial = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    if enable_partial_rollout:
                        # NOTE: Added by Reasoning360 partial rollout: parse the response.
                        is_partial = (
                            output.outputs[sample_id].finish_reason == "length"
                            and (
                                # TODO: this does not count original prompt ids.
                                # Need to consider the factor.
                                (
                                    len(output.prompt_token_ids) +
                                    len(output.outputs[sample_id].token_ids)
                                ) <
                                self.sampling_params.max_tokens
                            )
                        )
                        response_is_partial.append(is_partial)
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(
                response, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)
            response_is_partial = torch.tensor(response_is_partial, dtype=torch.bool, device=idx.device)

            if self.sampling_params.n > 1 and do_sample:
                # NOTE: Modified by Reasoning360. Only repeat raw prompts.
                if enable_partial_rollout:
                    repeats = torch.where(
                        prompts.batch['is_partial'].reshape(-1),
                        torch.tensor(1, device=idx.device),
                        torch.tensor(self.sampling_params.n, device=idx.device),
                    )
                    idx = _repeat_interleave(idx, repeats)
                    attention_mask = _repeat_interleave(attention_mask, repeats)
                    position_ids = _repeat_interleave(position_ids, repeats)
                    batch_size = torch.sum(repeats).cpu().item()
                else:
                    idx = _repeat_interleave(idx, self.sampling_params.n)
                    attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                    position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                    batch_size = batch_size * self.sampling_params.n
                    if 'multi_modal_inputs' in non_tensor_batch.keys():
                        non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                    self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device
        )
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, 3, -1
            )

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        tokens_per_second = torch.sum(response_attention_mask).item() / t()
        import os

        print(
            f'Tokens per second: {tokens_per_second} t/s on device {os.environ["CUDA_VISIBLE_DEVICES"]} on host {os.uname().nodename}',
            flush=True,
        )

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "is_partial": response_is_partial,  # NOTE: added by Reasoning360
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3")
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        # NOTE: added by Reasoning360
        metrics = self.report_memory_usage(reset=True)
        # NOTE: we do not use meta_info because dp collect fn only picks
        # meta_info of the first data.
        metrics = {
            'metrics_' + k: np.asarray([v] * seq.size(0), dtype=object) for k, v in metrics.items()
        }
        non_tensor_batch.update(metrics)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def report_memory_usage(self, reset: bool=False):
        # NOTE: added by Reasoning360
        method = getattr(self.inference_engine.llm_engine, 'report_page_usage_history', None)
        if method is not None:
            return method(reset=reset)
        return {}

    def _monkey_patch_vllm_engine_v0(self):
        """
        NOTE: added by Reasoning360.
        Monkey patching vllm engine v0 to support page usage report.
        """
        from vllm.v1.engine.llm_engine import LLMEngine as V1VllmEngine
        from vllm.engine.llm_engine import LLMEngine

        if isinstance(self.inference_engine.llm_engine, V1VllmEngine):
            # FIXME: monkey-patch V1 engine as well.
            return
        assert isinstance(self.inference_engine.llm_engine, LLMEngine)
        engine = self.inference_engine.llm_engine
        engine.max_page_usage = 0.
        engine.page_usage_average = 0.
        engine.page_usage_sample_times = 0

        step = engine.step

        def custom_step(self: LLMEngine):
            ret = step()
            # update page usage metrics
            num_total_gpu = self.cache_config.num_gpu_blocks
            gpu_cache_usage_sys = 0.
            if num_total_gpu is not None:
                num_free_gpu = sum(
                    scheduler.block_manager.get_num_free_gpu_blocks()
                    for scheduler in self.scheduler)
                gpu_cache_usage_sys = 1.0 - (num_free_gpu / num_total_gpu)
            self.max_page_usage = max(self.max_page_usage, gpu_cache_usage_sys)
            self.page_usage_average = (
                (self.page_usage_average * self.page_usage_sample_times + gpu_cache_usage_sys) /
                (self.page_usage_sample_times + 1))
            self.page_usage_sample_times += 1
            return ret

        def report_page_usage_history(self, reset=False):
            # NOTE: added by Reasoning360
            if reset:
                max_page_usage = self.max_page_usage
                page_usage_average = self.page_usage_average
                self.max_page_usage = 0
                self.page_usage_average = 0
                self.page_usage_sample_times = 0
                return {
                    "gpu_max_page_usage": max_page_usage,
                    "gpu_average_page_usage": page_usage_average,
                }
            return {
                "gpu_max_page_usage": self.max_page_usage,
                "gpu_average_page_usage": self.page_usage_average,
            }

        engine.step = types.MethodType(custom_step, engine)
        engine.report_page_usage_history = types.MethodType(
            report_page_usage_history, engine
        )
