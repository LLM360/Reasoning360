from datasets import load_dataset
from verl.utils.reward_score.math_llm_judge import match_answer
from tqdm import tqdm
import json
from collections import Counter
import multiprocessing as mp
import os

def load_batch_file(chunk_idx):
    """Load a single batch results file and return its contents."""
    results = []
    file_path = f'../batch_results/batch_results_chunk_{chunk_idx}.jsonl'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
    return results

def process_data_item(item):
    """Process a single data item and return processed information."""
    response = item['response']['body']['choices']['message']['content']
    id = item['custom_id']
    question_id = int(id.split("-")[1])
    sample_id = int(id.split("-")[-1])
    is_matched, extracted_model_output = match_answer(response)
    if not is_matched:
        extracted_model_output = "None"
    return question_id, extracted_model_output

def main():
    print("Loading dataset...")
    dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset",
                        trust_remote_code=True,
                        split='train')
    # get answer list from dataset
    answers = dataset['answer']
    # get question list from dataset
    problems = dataset['problem']

    print("Loading batch results...")
    # Determine the number of available CPU cores and create a process pool
    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)
    
    # Use multiprocessing to load batch files in parallel
    chunk_indices = range(0, 78)
    data = []
    results = []
    with tqdm(total=len(chunk_indices), desc="Loading chunks") as pbar:
        for chunk_result in pool.imap_unordered(load_batch_file, chunk_indices):
            data.extend(chunk_result)
            pbar.update(1)
    
    print(f"Loaded {len(data)} items from batch results.")
    print(f"Processing batch results using {num_cores} cores...")
    
    # Process the data in parallel
    processed_items = []
    with tqdm(total=len(data), desc="Processing items") as pbar:
        for result in pool.imap_unordered(process_data_item, data, chunksize=100):
            processed_items.append(result)
            pbar.update(1)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Organize the processed items into the question_output dictionary
    question_output = {}
    for question_id, extracted_model_output in processed_items:
        if question_id not in question_output:
            question_output[question_id] = []
        question_output[question_id].append(extracted_model_output)
    
    # turn list into a count dictionary
    question_output = {k: Counter(v) for k, v in question_output.items()}

    print("Saving question_output...")
    # save question_output
    with open('question_output_deepscale_r.json', 'w') as f:
        json.dump(question_output, f)

if __name__ == "__main__":
    # This is important for multiprocessing to work correctly
    main()
