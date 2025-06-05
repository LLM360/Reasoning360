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
Generate responses given a dataset of prompts
"""

import json
import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    # NOTE: added by Reasoning360
    if 'olmoe' in local_path.lower() and 'instruct' not in local_path.lower():
        tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    is_polars_df = False
    if 'livecodebench' in config.data.path:
        import polars as pl
        dataset = pl.read_parquet(config.data.path)
        chat_lst = list(dataset[config.data.prompt_key])
        chat_lst = [list(chat) for chat in chat_lst]
        ground_truth_lst = list(dataset["reward_model"])
        is_polars_df = True
    else:
        dataset = pd.read_parquet(config.data.path)
        chat_lst = dataset[config.data.prompt_key].tolist()
        chat_lst = [chat.tolist() for chat in chat_lst]
        ground_truth_lst = dataset["reward_model"].tolist()
    
    # NOTE: added by Reasoning360. handle n_samples
    if config.data.n_samples > 1:
        chat_lst = chat_lst * config.data.n_samples
        ground_truth_lst = ground_truth_lst * config.data.n_samples

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name="cuda" if is_cuda_available else "npu")
    wg.init_model()

    # NOTE: updated by Reasoning360. Sample n times together
    total_samples = len(chat_lst) # chat_lst is repeated
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dispatch_dp_size = wg.world_size
    num_batch = -(-total_samples // config_batch_size)

    output_lst = []

    # total_samples = len(dataset)
    # config_batch_size = config.data.batch_size
    # num_batch = -(-total_samples // config_batch_size)
    # output_lst = [[] for _ in range(config.data.n_samples)]

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        # NOTE: modified by Reasoning360. Sample n times altogether.
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        batch_size = data_padded.batch['input_ids'].shape[0]

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        # START TO GENERATE FOR 1 TIME SINCE WE'VE ALREADY HANDLED n_samples beforehand
        output_padded = wg.generate_sequences(data_padded)
        # remove dummy data
        output = unpad_dataproto(output_padded, pad_size=pad_size)
        output_texts = []
        for i in range(len(output)):
            data_item = output[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            # TODO: batch this operation.
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            output_texts.append(response_str)

        # remove the padding
        pad_token = tokenizer.pad_token
        output_text_unpad = []
        for text in output_texts:
            output_text_unpad.append(text.replace(pad_token, ''))

        output_lst.extend(output_text_unpad)

    # convert output_lst from (n_samples * n_data ,) to (n_data, n_sampels)
    original_data_size = len(dataset)
    output_lst = np.array(output_lst).reshape(config.data.n_samples, original_data_size)
    output_lst = output_lst.T.tolist()
    
    original_chat_lst = chat_lst[:original_data_size]
    original_ground_truth_lst = ground_truth_lst[:original_data_size]

    # add to the data frame
    if is_polars_df:
        dataset = dataset.with_columns(pl.Series("responses", output_lst))
        # write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.write_parquet(config.data.output_path)
    else:
        # For pandas, use standard bracket assignment
        dataset['responses'] = output_lst
        # write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(config.data.output_path)
    # NOTE: added by Reasoning360. dump results
    result_list = [
        {
            "prompt": chat,
            "response": output,
            "ground_truth": str(ground_truth),
        } 
        for chat, output, ground_truth in zip(original_chat_lst, output_lst, original_ground_truth_lst)
    ]
    model_name = config.model.path.split('/')[-1]
    with open(config.data.output_path.replace('.parquet', f'_{model_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
