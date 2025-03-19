from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from multiprocessing import Pool


class NaiveParallelRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_metric = kwargs.get("reward_metric", None)

    def __call__(self, data: DataProto):

        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        to_print = {}
        for i in range(len(data)):
            data_source = data[i].non_tensor_batch["data_source"]
            if data_source not in to_print:
                to_print[data_source] = []
            if len(to_print[data_source]) < self.num_examine:
                to_print[data_source].append(i)

        def process_item(i):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch["data_source"]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                reward_metric=self.reward_metric,
            )
            position = valid_response_length - 1

            if i in to_print.get(data_source, []):
                return i, position, score, sequences_str
            return i, position, score, None

        with Pool() as pool:
            results = pool.map(process_item, range(len(data)))

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i, position, score, seq in results:
            reward_tensor[i, position] = score
            if seq is not None:
                print(seq)

        return reward_tensor
