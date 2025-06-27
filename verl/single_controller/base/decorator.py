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

import inspect
from functools import wraps
from types import FunctionType
from typing import Dict, List, Tuple

import torch

from verl.protocol import DataProtoFuture, _padding_size_key
from verl.utils.py_functional import DynamicEnum

# here we add a magic number of avoid user-defined function already have this attribute
MAGIC_ATTR = "attrs_3141562937"


class Dispatch(DynamicEnum):
    """Enum class defining different dispatch modes for distributed computation.

    Each mode represents a specific strategy for distributing data across
    different ranks in a distributed system. The modes are used to control
    how data is partitioned and processed across different worker groups.
    """

    _registry = {}
    _next_value = 0


def init_predefined_dispatch_mode():
    Dispatch.register("RANK_ZERO")
    Dispatch.register("ONE_TO_ALL")
    Dispatch.register("ALL_TO_ALL")
    Dispatch.register("MEGATRON_COMPUTE")
    Dispatch.register("MEGATRON_PP_AS_DP")
    Dispatch.register("MEGATRON_PP_ONLY")
    Dispatch.register("MEGATRON_COMPUTE_PROTO")
    Dispatch.register("MEGATRON_PP_AS_DP_PROTO")
    Dispatch.register("DP_COMPUTE")
    Dispatch.register("DP_COMPUTE_PROTO")
    Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
    Dispatch.register("DP_COMPUTE_METRIC")
    Dispatch.register("MEGATRON_PP_DUMMY_PROTO")
    # This is a special dispatch mode for vllm ExternalRayDistributedExecutor
    Dispatch.register("DIRECT_ROLLOUT_METHOD")


class Execute(DynamicEnum):
    """Enum class defining different execution modes for distributed computation.

    These modes control how a function should be executed across different ranks
    in a distributed system.
    """

    _registry = {}
    _next_value = 0


def init_predefined_execute_mode():
    Execute.register("ALL")
    Execute.register("RANK_ZERO")


# Initialize the two Dynamic Enum Classes
init_predefined_dispatch_mode()
init_predefined_execute_mode()


def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    from verl.protocol import DataProto, DataProtoFuture

    splitted_args = []
    for arg in args:
        assert isinstance(arg, (DataProto, DataProtoFuture))
        splitted_args.append(arg.chunk(chunks=chunks))

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert isinstance(val, (DataProto, DataProtoFuture))
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs


def _split_args_kwargs_data_proto_with_auto_padding(chunks, *args, **kwargs):
    from verl.protocol import DataProto, DataProtoFuture

    splitted_args = []
    splitted_kwargs = {}

    data_proto_len = None
    padding_size = None
    for arg in args:
        assert isinstance(arg, (DataProto, DataProtoFuture))
        if isinstance(arg, DataProto) and arg.is_padding_enabled():
            # for padding, we only support DataProto with same length
            if data_proto_len is None:
                data_proto_len = len(arg)
                padding_size = (chunks - (data_proto_len % chunks)) if (data_proto_len % chunks > 0) else 0
                splitted_kwargs[_padding_size_key] = padding_size
            else:
                assert data_proto_len == len(arg), f"expecting all arg share same length of {data_proto_len}, but got {len(arg)}"
                data_proto_len = len(arg)
            arg.padding(padding_size=padding_size)

        splitted_args.append(arg.chunk(chunks=chunks))

    for key, val in kwargs.items():
        assert isinstance(val, (DataProto, DataProtoFuture))
        if isinstance(val, DataProto) and val.is_padding_enabled():
            # for padding, we only support DataProto with same length
            if data_proto_len is None:
                data_proto_len = len(val)
                padding_size = chunks - (data_proto_len % chunks)
                splitted_kwargs[_padding_size_key] = padding_size
            else:
                assert data_proto_len == len(val), f"expecting all arg share same length of {data_proto_len}, but got {len(val)}"
                data_proto_len = len(val)
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dummy_direct_rollout_call(worker_group, *args, **kwargs):
    raise NotImplementedError("Direct rollout call is forbidden.")


def dispatch_all_to_all(worker_group, *args, **kwargs):
    return args, kwargs


def collect_all_to_all(worker_group, output):
    return output


def dispatch_megatron_compute(worker_group, *args, **kwargs):
    """
    User passes in dp data. The data is dispatched to all tp/pp ranks with the same dp
    """
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    assert isinstance(worker_group, MegatronWorkerGroup), f"worker_group must be MegatronWorkerGroup, Got {type(worker_group)}"

    all_args = []
    for arg in args:
        assert isinstance(arg, (Tuple, List)) and len(arg) == worker_group.dp_size
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            transformed_args.append(arg[local_dp_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        assert isinstance(v, (Tuple, List)) and len(v) == worker_group.dp_size
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            transformed_v.append(v[local_dp_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_megatron_compute(worker_group, output):
    """
    Only collect the data from the tp=0 and pp=last and every dp ranks
    """
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    assert isinstance(worker_group, MegatronWorkerGroup)
    output_in_dp = []
    pp_size = worker_group.get_megatron_global_info().pp_size
    for global_rank in range(worker_group.world_size):
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.pp_rank == pp_size - 1 and local_rank_info.cp_rank == 0:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def dispatch_megatron_compute_data_proto(worker_group, *args, **kwargs):
    """
    All the args and kwargs must be DataProto. The batch will be chunked by dp_size and passed to each rank
    """
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    assert isinstance(worker_group, MegatronWorkerGroup)

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.dp_size, *args, **kwargs)
    return dispatch_megatron_compute(worker_group, *splitted_args, **splitted_kwargs)


def _concat_data_proto_or_future(output: List):
    import ray

    from verl.protocol import DataProto, DataProtoFuture

    # make sure all the elements in output has the same type
    for o in output:
        assert type(o) is type(output[0])

    o = output[0]

    if isinstance(o, DataProto):
        return DataProto.concat(output)
    elif isinstance(o, ray.ObjectRef):
        return DataProtoFuture.concat(output)
    else:
        raise NotImplementedError


def collect_megatron_compute_data_proto(worker_group, output):
    """
    Each output must be a DataProto. We concat the dim=0 of output
    """
    import ray

    from verl.protocol import DataProto

    output = collect_megatron_compute(worker_group, output)
    for o in output:
        assert isinstance(o, (DataProto, ray.ObjectRef)), f"expecting {o} to be DataProto, but got {type(o)}"

    return _concat_data_proto_or_future(output)


def dispatch_megatron_pp_as_dp(worker_group, *args, **kwargs):
    """
    treat pp as dp.
    """
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    assert isinstance(worker_group, MegatronWorkerGroup)

    pp_size = worker_group.pp_size
    dp_size = worker_group.dp_size
    cp_size = worker_group.cp_size
    pp_dp_cp_size = pp_size * dp_size * cp_size

    all_args = []
    for arg in args:
        assert isinstance(arg, (List, Tuple)) and len(arg) == pp_dp_cp_size
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            local_pp_rank = worker_group.get_megatron_rank_info(rank=i).pp_rank
            local_cp_rank = worker_group.get_megatron_rank_info(rank=i).cp_rank
            # compute the rank in arg. Note that the order is dp then cp then pp
            # Also note that the outputs within a pp group will be firstly allgathered, then only the output of pp0 will be collected.
            # For pp=2 dp=4, a batch of data "ABCDEFGH" should be dispatched and collected in below order:
            #    dispatch:       pp_allgther:        collect:
            #   dp 0 1 2 3      dp  0  1  2  3
            # pp +---------+  pp +-------------+
            #  0 | A C E G |   0 | AB CD EF GH |     ABCDEFGH
            #  1 | B D F H |   1 | AB CD EF GH |
            #    +---------+     +-------------+
            dp_cp_rank = local_cp_rank * dp_size + local_dp_rank
            arg_rank = dp_cp_rank * pp_size + local_pp_rank

            transformed_args.append(arg[arg_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        assert isinstance(v, (List, Tuple)) and len(v) == pp_dp_cp_size, f"expect len(v)=={pp_dp_cp_size}, got {len(v)}"
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            local_pp_rank = worker_group.get_megatron_rank_info(rank=i).pp_rank
            local_cp_rank = worker_group.get_megatron_rank_info(rank=i).cp_rank
            # compute the rank in arg. Note that the order is dp then cp then pp
            dp_cp_rank = local_cp_rank * dp_size + local_dp_rank
            arg_rank = dp_cp_rank * pp_size + local_pp_rank
            transformed_v.append(v[arg_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_megatron_pp_as_dp(worker_group, output):
    """
    treat pp as dp. Only collect data on tp=0
    """
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    assert isinstance(worker_group, MegatronWorkerGroup)
    output_in_dp = []
    for global_rank in range(worker_group.world_size):
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def collect_megatron_pp_only(worker_group, output):
    """
    Only collect output of megatron pp. This is useful when examine weight names as they are identical in tp/dp
    """
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    assert isinstance(worker_group, MegatronWorkerGroup)
    output_in_pp = []
    for global_rank in range(worker_group.world_size):
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.dp_rank == 0:
            output_in_pp.append(output[global_rank])
    return output_in_pp


def dispatch_megatron_pp_as_dp_data_proto(worker_group, *args, **kwargs):
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    assert isinstance(worker_group, MegatronWorkerGroup)

    pp_dp_cp_size = worker_group.dp_size * worker_group.pp_size * worker_group.cp_size
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(pp_dp_cp_size, *args, **kwargs)
    ret = dispatch_megatron_pp_as_dp(worker_group, *splitted_args, **splitted_kwargs)
    return ret


def collect_megatron_pp_as_dp_data_proto(worker_group, output):
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

    assert isinstance(worker_group, MegatronWorkerGroup)

    output = collect_megatron_pp_as_dp(worker_group, output)
    return _concat_data_proto_or_future(output)


def dispatch_dp_compute(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    for arg in args:
        assert isinstance(arg, (Tuple, List)) and len(arg) == worker_group.world_size
    for k, v in kwargs.items():
        assert isinstance(v, (Tuple, List)) and len(v) == worker_group.world_size
    return args, kwargs


def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return output


def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    # Note: enable auto padding for dp compute DatapProto
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs


def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert isinstance(args[0], FunctionType)  # NOTE: The first one args is a function!

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    return splitted_args_with_func, splitted_kwargs


def collect_dp_compute_data_proto(worker_group, output):
    import ray

    from verl.protocol import DataProto

    for o in output:
        assert isinstance(o, (DataProto, ray.ObjectRef)), f"expecting {o} to be DataProto, but got {type(o)}"

    output = collect_dp_compute(worker_group, output)
    return _concat_data_proto_or_future(output)


MAGIC_PREFIX = "__verl_dummy_tensor_"
def _materialize_dummy_data_proto(arg):
    from verl.protocol import DataProto
    from tensordict import TensorDict
    import numpy as np

    if not isinstance(arg, DataProto):
        return arg

    # This is not a dummy data proto
    if not arg.meta_info.get(f"{MAGIC_PREFIX}is_dummy", False):
        return arg
    arg.meta_info.pop(f"{MAGIC_PREFIX}is_dummy")

    new_batch = {}
    new_non_tensor_batch = {}
    batch_size = None
    for k, v in arg.batch.items():
        assert f"{MAGIC_PREFIX}batch_{k}_shape" in arg.meta_info
        shape = arg.meta_info[f"{MAGIC_PREFIX}batch_{k}_shape"]
        new_batch[k] = torch.zeros(shape, dtype=v.dtype, device=v.device)
        arg.meta_info.pop(f"{MAGIC_PREFIX}batch_{k}_shape")
        batch_size = batch_size or shape[0]
        assert batch_size == shape[0], f"{batch_size=}, {shape=}"
    for k, v in arg.non_tensor_batch.items():
        assert f"{MAGIC_PREFIX}non_tensor_batch_{k}_shape" in arg.meta_info
        shape = arg.meta_info[f"{MAGIC_PREFIX}non_tensor_batch_{k}_shape"]
        new_non_tensor_batch[k] = np.zeros(shape, dtype=v.dtype)
        arg.meta_info.pop(f"{MAGIC_PREFIX}non_tensor_batch_{k}_shape")
        assert batch_size == shape[0], f"{batch_size=}, {shape=}"
    return DataProto(
        batch=TensorDict(new_batch, batch_size=batch_size),
        non_tensor_batch=new_non_tensor_batch,
        meta_info=arg.meta_info,
    )


def _make_dummy_data_proto(arg):
    from verl.protocol import DataProto
    import numpy as np
    from tensordict import TensorDict

    if not isinstance(arg, DataProto):
        return arg

    new_batch = TensorDict({}, batch_size=[1])
    new_non_tensor_batch = {}
    meta_info = arg.meta_info.copy()

    empty_shape = [1]
    for k, v in arg.batch.items():
        shape = v.shape
        # empty_shape = [0] + list(shape[1:])
        new_batch[k] = torch.zeros(empty_shape, dtype=v.dtype, device=v.device)
        meta_info[f"{MAGIC_PREFIX}batch_{k}_shape"] = shape

    for k, v in arg.non_tensor_batch.items():
        shape = v.shape
        # empty_shape = [0] + list(shape[1:])
        new_non_tensor_batch[k] = np.zeros(empty_shape, dtype=v.dtype)
        meta_info[f"{MAGIC_PREFIX}non_tensor_batch_{k}_shape"] = shape
    meta_info[f"{MAGIC_PREFIX}is_dummy"] = True
    return DataProto(batch=new_batch, non_tensor_batch=new_non_tensor_batch, meta_info=meta_info)


def dispatch_megatron_pp_dummy_data_proto(worker_group, *args, **kwargs):
    """
    NOTE: added by Reasoning360. It reads from a special keyword argument `verl_pp_send_rank: Sequence[int]`
    It handles other arguments the same as `dispatch_megatron_compute_data_proto`, but the DataProto args are different that:
    For Data Parallel Group (DP), the dispatch pattern is the same as `dispatch_megatron_compute_data_proto`.
    For Pipeline Parallel Group (PP), only workers with a PP rank within `verl_pp_send_rank` will be dispatched. Other workers
    wil receive an empty DataProto, with meta_info pairs of `batch_{key}_shape: value.shape for key, value in arg.batch`.
    NOTE: this function cannot handle DataProtoFuture now.
    TODO: broadcast within TP ranks after receiving, then TP ranks > 0 will also receive dummy data.
    """
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    from verl.protocol import DataProto, DataProtoFuture

    assert isinstance(worker_group, MegatronWorkerGroup)

    # Extract the special keyword argument for PP send ranks
    verl_pp_send_rank = kwargs.pop("verl_pp_send_rank", None)
    if verl_pp_send_rank is None:
        verl_pp_send_rank = (0, worker_group.pp_size - 1)

    # First, split the DataProto arguments by dp_size like in megatron_compute_data_proto
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.dp_size, *args, **kwargs)

    # Now apply the megatron compute dispatch pattern
    all_args, all_kwargs = dispatch_megatron_compute(worker_group, *splitted_args, **splitted_kwargs)

    # For each worker, check if it should receive data or empty DataProto
    for rank in range(worker_group.world_size):
        local_rank_info = worker_group.get_megatron_rank_info(rank=rank)
        pp_rank = local_rank_info.pp_rank
        tp_rank = local_rank_info.tp_rank

        # If this worker's PP rank is not in the send list, replace with empty DataProto
        if pp_rank not in verl_pp_send_rank or tp_rank != 0:
            # Create empty DataProto with shape information from original args
            for arg_idx, arg in enumerate(all_args):
                if isinstance(arg[rank], (DataProto, DataProtoFuture)):
                    # Get the original DataProto to extract shape information
                    original_arg = arg[rank]
                    if original_arg is not None and isinstance(original_arg, DataProto):
                        all_args[arg_idx][rank] = _make_dummy_data_proto(original_arg)

            # Handle kwargs similarly
            for key, val_list in all_kwargs.items():
                if isinstance(val_list[rank], (DataProto, DataProtoFuture)):
                    original_val = val_list[rank]
                    if original_val is not None and isinstance(original_val, DataProto):
                        all_kwargs[key][rank] = _make_dummy_data_proto(original_val)

    return all_args, all_kwargs


# Global registry for dispatch mode.
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.MEGATRON_COMPUTE: {
        "dispatch_fn": dispatch_megatron_compute,
        "collect_fn": collect_megatron_compute,
    },
    Dispatch.MEGATRON_PP_AS_DP: {
        "dispatch_fn": dispatch_megatron_pp_as_dp,
        "collect_fn": collect_megatron_pp_as_dp,
    },
    Dispatch.MEGATRON_PP_ONLY: {"dispatch_fn": dispatch_one_to_all, "collect_fn": collect_megatron_pp_only},
    Dispatch.MEGATRON_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_megatron_compute_data_proto,
        "collect_fn": collect_megatron_compute_data_proto,
    },
    Dispatch.MEGATRON_PP_AS_DP_PROTO: {
        "dispatch_fn": dispatch_megatron_pp_as_dp_data_proto,
        "collect_fn": collect_megatron_pp_as_dp_data_proto,
    },
    Dispatch.DP_COMPUTE: {"dispatch_fn": dispatch_dp_compute, "collect_fn": collect_dp_compute},
    Dispatch.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
        "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_METRIC: {"dispatch_fn": dispatch_dp_compute_data_proto, "collect_fn": collect_dp_compute},
    Dispatch.DIRECT_ROLLOUT_METHOD: {
        "dispatch_fn": dummy_direct_rollout_call,
        "collect_fn": dummy_direct_rollout_call,
    },
    Dispatch.MEGATRON_PP_DUMMY_PROTO: {
        "dispatch_fn": dispatch_megatron_pp_dummy_data_proto,
        "collect_fn": collect_megatron_compute_data_proto,
    },
}


def get_predefined_dispatch_fn(dispatch_mode):
    return DISPATCH_MODE_FN_REGISTRY[dispatch_mode]


def register_dispatch_mode(dispatch_mode_name, dispatch_fn, collect_fn):
    """
    Register a new dispatch mode.
    """
    dispatch_mode = Dispatch.register(dispatch_mode_name)
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode not in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode_name {dispatch_mode_name} already exists"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


def update_dispatch_mode(dispatch_mode, dispatch_fn, collect_fn):
    """
    Update the dispatch mode.
    """
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode {dispatch_mode} not found"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


def get_predefined_execute_fn(execute_mode):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    predefined_execute_mode_fn = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    return predefined_execute_mode_fn[execute_mode]


def _check_dispatch_mode(dispatch_mode):
    assert isinstance(dispatch_mode, (Dispatch, Dict)), f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    if isinstance(dispatch_mode, Dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode):
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


def _materialize_futures(*args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        # add more type to materialize
        new_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, DataProtoFuture):
            kwargs[k] = v.get()

    new_args = tuple(new_args)
    return new_args, kwargs


def _materialize_dummy(*args, **kwargs):
    from verl.protocol import DataProto

    new_args = []
    for arg in args:
        if isinstance(arg, DataProto):
            arg = _materialize_dummy_data_proto(arg)
        new_args.append(arg)
    for k in kwargs:
        if isinstance(kwargs[k], DataProto):
            kwargs[k] = _materialize_dummy_data_proto(kwargs[k])
    return tuple(new_args), kwargs


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    """Register a function with distributed execution configuration.

    This decorator registers a function with specific dispatch and execution modes
    for distributed computation. It handles both synchronous and asynchronous
    functions, and optionally materializes futures before execution.

    Args:
        dispatch_mode:
            Dispatch mode for computation distribution. Default: Dispatch.ALL_TO_ALL.
        execute_mode:
            Execute mode for computation distribution. Default: Execute.ALL.
        blocking:
            Whether the execution should be blocking. Defaults to True.
        materialize_futures:
            Whether to materialize the data before dispatching. Defaults to True.
        materialize_dummy:
            Whether it receives a dummy DataProto. If so, it will materialize a dummy
            tensor based on the metadata in the DataProto. This is to receive unused
            data for intermediate ranks of pipeline parallel.

    Returns:
        A decorator that wraps the original function with distributed execution
        configuration.
    """
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    materialize_dummy = dispatch_mode == Dispatch.MEGATRON_PP_DUMMY_PROTO

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            if materialize_dummy:
                args, kwargs = _materialize_dummy(*args, **kwargs)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            if materialize_dummy:
                args, kwargs = _materialize_dummy(*args, **kwargs)
            return await func(*args, **kwargs)

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper

    return decorator
