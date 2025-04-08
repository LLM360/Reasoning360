"""
Note: Before running this script,
you should create the input.jsonl file with the following content:
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!  List 3 NBA players and tell a story"}],"max_tokens": 300}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an assistant. "},{"role": "user", "content": "Hello world! List three capital and tell a story"}],"max_tokens": 500}}
"""

import json
import os
import time
import argparse
import openai


class OpenAIBatchProcessor:
    def __init__(self, output_folder=".", server_url="http://127.0.0.1:30000/v1"):
        client = openai.Client(base_url=server_url, api_key="EMPTY")
        self.client = client
        self.chunk_size = 1024  # Default chunk size
        self.output_folder = output_folder
        
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def process_batch(self, input_file_path, endpoint, completion_window, chunk_size=None):
        if chunk_size:
            self.chunk_size = chunk_size
            
        # Load and replicate the input data
        replicated_data = self._replicate_requests(input_file_path)
        
        # Split data into chunks
        chunks = self._split_into_chunks(replicated_data)
        
        all_results = []
        
        # Process each chunk as a separate batch job
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} requests...")
            
            # Create a temporary file for this chunk
            temp_file_path = os.path.join(self.output_folder, f"temp_chunk_{i}.jsonl")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                for item in chunk:
                    f.write(json.dumps(item) + "\n")
            
            # Process this chunk
            chunk_results = self._process_single_batch(temp_file_path, endpoint, completion_window)
            
            if chunk_results:
                # Save chunk results to a separate file
                chunk_result_file = os.path.join(self.output_folder, f"batch_results_chunk_{i}.jsonl")
                with open(chunk_result_file, "w", encoding="utf-8") as f:
                    for result in chunk_results:
                        f.write(json.dumps(result) + "\n")
                
                all_results.extend(chunk_results)
                print(f"Saved results for chunk {i+1} to {chunk_result_file}")
            
            # Clean up temporary file
            os.remove(temp_file_path)
            
        return all_results

    def _replicate_requests(self, input_file_path):
        """Load requests from file and replicate each 64 times with modified IDs"""
        original_requests = []
        
        with open(input_file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    original_requests.append(json.loads(line))
        
        replicated_requests = []
        for req in original_requests:
            base_id = req.get("custom_id", "request")
            
            for i in range(32):
                # Create a deep copy of the original request
                new_req = json.loads(json.dumps(req))
                # Modify the ID
                new_req["custom_id"] = f"{base_id}-{i}"
                replicated_requests.append(new_req)
                
        print(f"Replicated {len(original_requests)} requests into {len(replicated_requests)} requests")
        return replicated_requests
    
    def _split_into_chunks(self, data):
        """Split data into chunks of specified size"""
        return [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
    
    def _process_single_batch(self, input_file_path, endpoint, completion_window):
        """Process a single batch job with the given input file"""
        # Upload the input file
        with open(input_file_path, "rb") as file:
            uploaded_file = self.client.files.create(file=file, purpose="batch")

        # Create the batch job
        batch_job = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint=endpoint,
            completion_window=completion_window,
        )

        print(f"Batch job created with ID: {batch_job.id}")
        print(f"Please monitor the running log of SGLang server.")

        # Monitor the batch job status
        while batch_job.status not in ["completed", "failed", "cancelled"]:
            time.sleep(5)
            batch_job = self.client.batches.retrieve(batch_job.id)

        # Check the batch job status and errors
        if batch_job.status == "failed":
            print(f"Batch job failed with status: {batch_job.status}")
            print(f"Batch job errors: {batch_job.errors}")
            return None

        # If the batch job is completed, process the results
        if batch_job.status == "completed":
            # print result of batch job
            print("batch", batch_job.request_counts)

            result_file_id = batch_job.output_file_id
            # Retrieve the file content from the server
            file_response = self.client.files.content(result_file_id)
            result_content = file_response.read()  # Read the content of the file

            # Save the content to a local file
            result_file_name = os.path.join(self.output_folder, f"temp_results_{batch_job.id}.jsonl")
            with open(result_file_name, "wb") as file:
                file.write(result_content)  # Write the binary content to the file
                
            # Load data from the saved JSONL file
            results = []
            with open(result_file_name, "r", encoding="utf-8") as file:
                for line in file:
                    json_object = json.loads(
                        line.strip()
                    )  # Parse each line as a JSON object
                    results.append(json_object)
            
            # Clean up temporary results file
            os.remove(result_file_name)
            
            return results
        else:
            print(f"Batch job failed with status: {batch_job.status}")
            return None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process batch inference requests")
    parser.add_argument("--input-file", type=str, default="deepscaler_problems.jsonl",
                        help="Path to the input JSONL file")
    parser.add_argument("--output-folder", type=str, default="./batch_results",
                        help="Folder to save output files")
    parser.add_argument("--chunk-size", type=int, default=1024,
                        help="Number of requests to process in each batch")
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions",
                        help="API endpoint to use")
    parser.add_argument("--completion-window", type=str, default="24h",
                        help="Completion window for the batch job")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:30000/v1",
                        help="Full URL for the SGLang server")
    
    args = parser.parse_args()
    
    # Initialize the OpenAIBatchProcessor with the output folder and server URL
    processor = OpenAIBatchProcessor(output_folder=args.output_folder, server_url=args.server_url)
    
    # Process the batch job with the specified parameters
    results = processor.process_batch(
        args.input_file, 
        args.endpoint, 
        args.completion_window, 
        chunk_size=args.chunk_size
    )
    
    # Print summary of results
    print(f"Total results processed: {len(results) if results else 0}")


if __name__ == "__main__":
    main()