import asyncio
import pandas as pd
from utils import LLMClient

# Hyperparameters - These can be modified by users
TIMEOUT_SECONDS = 24000  # 400 minutes
MAX_WINDOW_SIZE_PER_INSTANCE = 256    # Maximum concurrent requests per server instance
NUM_GENERATIONS = 16     # Number of generations per input
MODEL = "deepseek-ai/DeepSeek-R1-0528" # "Qwen/Qwen3-32B"

def main():
    """
    Process a dataset of inputs through the LLM and save responses.
    """
    # Initialize LLM client with user-defined parameters
    llm_client = LLMClient(
        timeout_seconds=TIMEOUT_SECONDS,
        max_window_size_per_instance=MAX_WINDOW_SIZE_PER_INSTANCE
    )

    # Read input data
    data_df = pd.read_json("~/libra-eval/libra_eval/datasets/iq_250.jsonl", lines=True)

    # Convert inputs to list for async processing
    inputs_list = data_df["messages"].tolist()

    # Process all inputs asynchronously with rate limiting
    responses = asyncio.run(llm_client.process_all_inputs(
        inputs_list, 
        num_generations=NUM_GENERATIONS,
        model=MODEL
    ))

    # Add responses to the DataFrame
    data_df['responses'] = responses
    
    # Save to JSONL file
    output_file = "llm_responses.jsonl"
    data_df.to_json(output_file, orient='records', lines=True)

    # Calculate success rate
    successful_generations = sum(len([r for r in resp if r is not None]) for resp in responses)
    total_expected_generations = len(inputs_list) * NUM_GENERATIONS

    print(f"Responses saved to {output_file}")
    print(f"Processed {successful_generations} out of {total_expected_generations} generations successfully")


if __name__ == "__main__":
    main()
