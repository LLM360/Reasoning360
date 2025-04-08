from datasets import load_dataset
import json
import math
import os
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from verl.utils.reward_score.math_llm_judge import grade_answer, math_equal, llm_check_answer
import time

# Load the dataset
dataset = load_dataset("SynthLabsAI/Big-Math-RL-Verified",
                      trust_remote_code=True,
                      split='train')

# Load output counts
output_count = json.load(open("question_output_big-math-rl-verified_problems.json"))

# Extract questions and answers from dataset
questions = {}
answers = {}
for i, item in enumerate(dataset):
    question_id = i
    questions[question_id] = item['problem']
    answers[question_id] = item['answer']

def compute_score_with_output(extracted_model_output, ground_truth, question):
    # grade simple algebra questions. if succeeded, return; otherwise, proceed to more complex grading
    if extracted_model_output == "None":
        return False, False, extracted_model_output
    
    if grade_answer(extracted_model_output, ground_truth):
        return True, False, extracted_model_output

    try:
        if "\\pi" in extracted_model_output or "\\pi" in ground_truth:
            equivs = []
            for pi in [math.pi, 3.14]:
                equivs.append(math_equal(extracted_model_output, ground_truth, timeout=True, pi=pi))
            is_correct = any(equivs)
        else:
            is_correct = math_equal(extracted_model_output, ground_truth, timeout=True)
    except:
        is_correct = False
    
    llm_correct = False
    # # Fixed issue: is_matched was undefined, using is_correct instead
    if is_correct is False:  # Only proceed if not already correct
        # use llm to check if the answer is correct
        try:
            is_correct = float(ground_truth) == float(extracted_model_output)
        except:
            is_correct = llm_check_answer(extracted_model_output, ground_truth, question)
            llm_correct = is_correct
    return is_correct, llm_correct, extracted_model_output

def process_question(item):
    """Process a single question and its outputs"""
    question_id_str, output_counts_for_question, question, answer = item
    question_id = int(question_id_str)
    
    saved_outputs = []
    
    for output, count in output_counts_for_question.items():
        if output == "None":
            saved_outputs.append(("None", count, False, False))
        else:
            try:
                # print("trying to compute score with output {} and answer {}".format(output, answer))
                is_correct, llm_correct, _ = compute_score_with_output(
                    output, 
                    answer, 
                    question
                )
                saved_outputs.append((output, count, is_correct, llm_correct))
            except Exception as e:
                # Handle potential errors
                print(f"Error processing question {question_id}, output {output}: {str(e)}")
                saved_outputs.append((output, count, False, False))
    
    return question_id, saved_outputs

def save_checkpoint(results, checkpoint_file="big-math-rl-verified-checkpoint.json"):
    """Save progress checkpoint to resume later if needed"""
    stats = {question_id: outputs for question_id, outputs in results}
    
    # Calculate checkpoint statistics
    correct_outputs = 0
    total_outputs = 0
    llm_correct_outputs = 0
    for question_id, outputs in stats.items():
        for _, count, is_correct, llm_correct in outputs:
            if is_correct:
                correct_outputs += count
            if llm_correct:
                llm_correct_outputs += count
            total_outputs += count
    
    accuracy = correct_outputs / total_outputs if total_outputs > 0 else 0
    llm_accuracy = llm_correct_outputs / total_outputs if total_outputs > 0 else 0
    checkpoint_data = {
        "stats": stats,
        "summary": {
            "correct": correct_outputs,
            "total": total_outputs,
            "accuracy": accuracy,
            "llm_accuracy": llm_accuracy,
            "processed_questions": len(stats),
            "timestamp": time.time()
        }
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"Checkpoint saved to {checkpoint_file} - Processed {len(stats)} questions.")

def load_checkpoint(checkpoint_file="big-math-rl-verified-checkpoint.json"):
    """Load checkpoint if it exists"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            print(f"Loaded checkpoint with {len(checkpoint_data['stats'])} processed questions.")
            print(f"Checkpoint statistics: {checkpoint_data['summary']}")
            
            # Convert question_id keys from strings back to integers if needed
            stats = {int(question_id): outputs for question_id, outputs in checkpoint_data['stats'].items()}
            return stats
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return {}
    else:
        print("No checkpoint found. Starting from scratch.")
        return {}

def main():
    # Use a reasonable number of processes
    num_processes = 100
    print(f"Using {num_processes} processes")
    
    # Set checkpoint interval (save every N questions processed)
    checkpoint_interval = 1000
    checkpoint_file = "big-math-rl-verified-checkpoint.json"
    
    # Load checkpoint if it exists
    processed_results = load_checkpoint(checkpoint_file)
    processed_question_ids = set(processed_results.keys())
    
    # Create a pool of workers
    pool = mp.Pool(processes=num_processes)

    # Prepare data for each task - only pass what's needed for each question
    tasks = []
    for question_id_str in output_count.keys():
        question_id = int(question_id_str)
        
        # Skip already processed questions when resuming from checkpoint
        if question_id in processed_question_ids:
            continue
            
        # Only pass the specific question and answer needed
        tasks.append((
            question_id_str,
            output_count[question_id_str],
            questions[question_id],
            answers[question_id]
        ))
    
    print(f"Processing {len(tasks)} remaining questions out of {len(output_count)} total questions")
    
    # Process questions in parallel with progress bar
    results = []
    
    # Initialize counter for checkpoint saving
    questions_since_checkpoint = 0
    
    # Process using imap_unordered for better performance with progress tracking
    with tqdm(total=len(tasks), desc="Processing questions") as pbar:
        for result in pool.imap_unordered(process_question, tasks, chunksize=1):
            results.append(result)
            pbar.update(1)
            
            questions_since_checkpoint += 1
            
            # Save checkpoint periodically
            if questions_since_checkpoint >= checkpoint_interval:
                # Combine new results with previously processed results
                all_results = list(processed_results.items()) + results
                save_checkpoint(all_results, checkpoint_file)
                questions_since_checkpoint = 0
    
    # Close the pool
    pool.close()
    pool.join()
    
    
    # # without multiprocessing
    # for question_id_str, output_counts_for_question, question, answer in tasks:
    #     question_id = int(question_id_str)
    #     print("shibo-index", question_id)
    #     results.append(process_question((question_id_str, output_counts_for_question, question, answer)))
    
    # Combine results with previously processed results from checkpoint if any
    all_results = list(processed_results.items()) + results
    
    # Combine results
    stats = {question_id: outputs for question_id, outputs in all_results}
    
    # Calculate overall statistics
    correct_outputs = 0
    total_outputs = 0
    llm_correct_outputs = 0
    for question_id, outputs in stats.items():
        for _, count, is_correct, llm_correct in outputs:
            if is_correct:
                correct_outputs += count
            if llm_correct:
                llm_correct_outputs += count
            total_outputs += count
    
    accuracy = correct_outputs / total_outputs if total_outputs > 0 else 0
    llm_accuracy = llm_correct_outputs / total_outputs if total_outputs > 0 else 0
    # Save the final results
    output_file = "big-math-rl-verified-stats.json"
    with open(output_file, 'w') as f:
        json.dump({
            "stats": stats,
            "summary": {
                "correct": correct_outputs,
                "total": total_outputs,
                "accuracy": accuracy,
                "llm_accuracy": llm_accuracy
            }
        }, f, indent=2)
    
    # Save final checkpoint
    save_checkpoint(all_results, checkpoint_file)
    
    print(f"Results saved to {output_file}")
    print(f"Accuracy: {accuracy:.4f} ({correct_outputs}/{total_outputs})")
    print(f"LLM Accuracy: {llm_accuracy:.4f} ({llm_correct_outputs}/{total_outputs})")
if __name__ == "__main__":
    main()