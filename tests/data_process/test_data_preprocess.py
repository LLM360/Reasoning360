import random
import transformers
import pytest
import logging
import datasets

from verl.utils.data_process.filter import LengthFilter
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(42)

MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-7B"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

cache_dir = datasets.config.HF_DATASETS_CACHE

EXTRA_INST_MAP = {
    "math": "Please output the final answer within \\boxed{}.",
    "tablereason": "Please output the final answer within \\boxed{}.",
    "simulation": "Please output the final answer within \\boxed{}.",
}

DATASETS_CONFIG = [
    # math
    {"domain": "math", "name": "bigmath_preview_filtered_mar21", "check_train": True, "check_test": False, "max_length": "unlimited"},  # -1 means no length filter applied
    {"domain": "math", "name": "deepscaler_preview", "check_train": True, "check_test": False, "max_length": "unlimited"},
    # # code
    {"domain": "codegen", "name": "leetcode2k", "check_train": False, "check_test": True, "max_length": 4096},
    {"domain": "codegen", "name": "primeintellect", "check_train": True, "check_test": False, "max_length": 4096},
    {"domain": "codegen", "name": "taco", "check_train": True, "check_test": False, "max_length": 4096},
    {"domain": "codegen", "name": "livecodebench", "check_train": True, "check_test": True, "max_length": 4096},
    {"domain": "codegen", "name": "humaneval", "check_train": False, "check_test": True, "max_length": 4096},
    {"domain": "codegen", "name": "mbpp", "check_train": True, "check_test": True, "max_length": 4096},
    # simulation
    {"domain": "simulation", "name": "codeio", "check_train": True, "check_test": True, "max_length": 4096},
    # table
    {"domain": "table", "name": "multihier", "check_train": True, "check_test": True, "max_length": 4096},
    
    # Add more datasets here as needed
]

@pytest.fixture(scope="module")
def tokenizer_fixture():
    return transformers.AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

def load_samples_to_check(data_domain: str, dataname: str, check_train: bool, check_test: bool, samples_to_check: int = 50):
    """
    Load samples to check
    """
    data_source = f"{data_domain}__{dataname}"
    logger.info(f"Loading {samples_to_check} samples from {data_source}")
    import importlib
    
    # if dataname == "bigmath_preview_filtered_mar21":
    module_path = f"data_preprocess.{data_domain}.{dataname}"
    module = importlib.import_module(module_path)
    train_dataset, test_dataset = module.get_datasets(cache_dir)
    
    result_datasets = {}
    if check_train:
        train_dataset = train_dataset.select(random.sample(range(len(train_dataset)), samples_to_check))
        result_datasets["train"] = train_dataset.map(module.make_map_fn("train", data_source), with_indices=True)
    if check_test:
        test_dataset = test_dataset.select(random.sample(range(len(test_dataset)), samples_to_check))
        result_datasets["test"] = test_dataset.map(module.make_map_fn("test", data_source), with_indices=True)
    return result_datasets


def check_length(data_entry, tokenizer, min_length, max_length, length_tolerance=100):
    """
    Check the length of the prompt
    """
    if max_length == "unlimited":
        max_length = 128000
        
    if "prompt" in data_entry and data_entry["prompt"]:
        prompt_tokens = tokenizer.tokenize(tokenizer.apply_chat_template(data_entry["prompt"], tokenize=False))
    elif "raw_prompt" in data_entry and data_entry["raw_prompt"]:
        prompt_tokens = tokenizer.tokenize(data_entry["raw_prompt"])
    else:
        raise ValueError("No prompt found in data")
    
    token_length = len(prompt_tokens)
    assert min_length <= token_length <= max_length - length_tolerance, \
        f"Token length {token_length} outside acceptable range [{min_length}, {max_length - length_tolerance}]"

def check_data_source_format(data_entry, data_domain, data_name):
    """
    Check the format of the data source
    """
    assert "data_source" in data_entry, "Missing data_source in extra_info"
    
    data_source = data_entry["data_source"]
    assert data_source is not None, "data_source is None"
    assert "__" in data_source, f"Invalid data_source format: {data_source}"
    
    domain_in_data_source, name_in_data_source = data_source.split("__")
    assert domain_in_data_source == data_domain, \
        f"Domain mismatch: {domain_in_data_source} != {data_domain}"
    assert name_in_data_source == data_name, \
        f"Name mismatch: {name_in_data_source} != {data_name}"


def check_prompt_format(data_entry, extra_instruction):
    """
    Check the format of the prompt
    """
    if extra_instruction is not None:
        raw_prompt = data_entry.get("raw_prompt")
        assert raw_prompt is not None, "Missing raw_prompt in data entry"
        assert extra_instruction in raw_prompt, \
            f"Extra instruction '{extra_instruction}' not found in prompt"


def check_special_tokens(data_entry, model_name_or_path):
    """
    Check the special tokens in the prompt
    """
    prompt = data_entry.get("raw_prompt")
    assert prompt is not None, "Missing raw_prompt in data entry"
    
    logger.info(prompt)
    if "Qwen" in model_name_or_path:
        assert "<|im_start|>" in prompt, \
            "Missing <|im_start|> token in prompt for Qwen model"
        assert "<|im_end|>" in prompt, \
            "Missing <|im_end|> token in prompt for Qwen model"
    
    assert "<think>" in prompt, "Missing <think> token in prompt"

@pytest.mark.parametrize("dataset_config", DATASETS_CONFIG)
def test_dataset_format(dataset_config, tokenizer_fixture):
    """
    Parameterized test for checking dataset format
    """
    dataname = dataset_config["name"]
    data_domain = dataset_config["domain"]
    check_train = dataset_config["check_train"]
    check_test = dataset_config["check_test"]
    min_length = dataset_config.get("min_length", 20)
    max_length = dataset_config.get("max_length", 2048)
    
    logger.info(f"Testing dataset: {dataname}")
    logger.info(f"Max length: {max_length}")
    datasets = load_samples_to_check(
        data_domain=data_domain, 
        dataname=dataname, 
        check_train=check_train, 
        check_test=check_test, 
        samples_to_check=50
    )
    
    for split, dataset in datasets.items():
        if dataset is None:
            continue
        # Skip samples that failed to pass the coding unittests, which return empty entries
        dataset = dataset.filter(lambda x: x["raw_prompt"] is not None and x["prompt"] is not None)
        
        # Filter by length, which is done in individual script
        if max_length != "unlimited":
            length_filter = LengthFilter(tokenizer=tokenizer_fixture, max_length=max_length)
            dataset = dataset.filter(lambda x: length_filter.check(x))
            
        logger.info(f"Testing {split} dataset: {data_domain}__{dataname}")
        for i, data_entry in enumerate(dataset):
            logger.debug(f"Checking sample {i+1}/{len(dataset)}")
            
            check_length(data_entry, tokenizer_fixture, min_length, max_length)
            check_data_source_format(data_entry, data_domain, dataname)
            
            extra_instruction = EXTRA_INST_MAP.get(data_domain, None)
            check_prompt_format(data_entry, extra_instruction)
            
            check_special_tokens(data_entry, MODEL_NAME_OR_PATH)
        
        logger.info(f"Successfully tested {len(dataset)} samples from {dataname}")