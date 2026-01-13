# Copyright 2025 Meituan Ltd. and/or its affiliates
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
Reasoning360 customizations for fully async policy training.

This module extends verl's experimental fully_async_policy with custom:
- Metrics (difficulty histograms)
- Reward computation (async signature with config/tokenizer)
- Dataset utilities (reasoning360-specific loaders)
- Resource pool management
"""

from reasoning360.experimental.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from reasoning360.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from reasoning360.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter

# Re-export verl components that don't need customization
from verl.experimental.fully_async_policy.param_sync import ParameterSynchronizer
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.fully_async_policy.fsdp2_utils import fsdp2_sharded_save_to_cpu, fsdp2_sharded_load_from_cpu
from verl.experimental.fully_async_policy.vllm_rollout.vllm_async_server import (
    FullyAsyncvLLMReplica,
    vLLMHttpServerForPartial,
)

__all__ = [
    "FullyAsyncRayPPOTrainer",
    "FullyAsyncTrainer",
    "FullyAsyncRollouter",
    "ParameterSynchronizer",
    "MessageQueue",
    "MessageQueueClient",
    "fsdp2_sharded_save_to_cpu",
    "fsdp2_sharded_load_from_cpu",
    "FullyAsyncvLLMReplica",
    "vLLMHttpServerForPartial",
]
