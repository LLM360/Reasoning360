import streamlit as st
import json
import os
import re
from abc import ABC, abstractmethod
import argparse
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from datasets import load_dataset

# Abstract base class for data sources
class DataSource(ABC):
    @abstractmethod
    def load_data(self):
        """Load and return data in a standardized format"""
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return the name of this data source"""
        pass

# Concrete implementation for batch results JSONL files
class BatchResultsSource(DataSource):
    def __init__(self, results_file, dataset_name="agentica-org/DeepScaleR-Preview-Dataset"):
        self.results_file = results_file
        self.dataset_name = dataset_name
    
    @property
    def name(self):
        return f"Batch Results: {os.path.basename(self.results_file)}"
    
    def load_data(self):
        # Load the batch results
        with open(self.results_file, 'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        
        # Load the dataset
        dataset = load_dataset(self.dataset_name, trust_remote_code=True, split='train')
        problems = dataset["problem"]
        answers = dataset["answer"]
        
        # Process the data
        processed_data = {}
        for d in data:
            response = d['response']['body']['choices']['message']['content']
            id = d['custom_id']  # custom_id is like 1-1, 1-2, 1-3, etc.
            question_id = int(id.split("-")[1])
            sample_id = int(id.split("-")[-1])
            
            if question_id not in processed_data:
                processed_data[question_id] = {
                    'problem': problems[question_id],
                    'reference_answer': answers[question_id],
                    'responses': []
                }
            
            processed_data[question_id]['responses'].append({
                'sample_id': sample_id,
                'full_response': response,
                'matched': d.get('matched', False),
                'model_output': "NOT_MATCHED" if not d.get('matched', False) else d.get('model_output', ''),
                'correct': d.get('correct', False)
            })
        
        return processed_data

# Helper functions
def process_latex(text):
    """Convert LaTeX delimiters to ones that Streamlit can render."""
    # Convert \( \) to $ $
    text = text.replace('\\(', '$').replace('\\)', '$')
    # Convert \[ \] to $$ $$
    text = text.replace('\\[', '$$').replace('\\]', '$$')
    
    return text

def plot_answer_distribution(responses):
    """Plot the distribution of model outputs"""
    model_outputs = [r.get('model_output', '') for r in responses]
    counter = Counter(model_outputs)
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame({
        'Answer': list(counter.keys()),
        'Count': list(counter.values())
    }).sort_values('Count', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df['Answer'], df['Count'])
    ax.set_xlabel('Model Output')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Model Outputs')
    
    # Rotate x-axis labels if there are many unique answers
    if len(df) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# Data viewer component
def display_data(processed_data):
    """Display the processed data in the Streamlit UI"""
    if not processed_data:
        st.info("No data available to display.")
        return
    
    # Add a dropdown to select the question
    question_ids = list(processed_data.keys())
    question_id = st.selectbox(
        "Select question ID", 
        question_ids,
        format_func=lambda x: f"Question {x} - Pass rate: {sum(r.get('correct', False) for r in processed_data[x]['responses']) / len(processed_data[x]['responses']):.2%} - Answer: {processed_data[x]['reference_answer']}"
    )
    
    if question_id:
        question_data = processed_data[question_id]
        
        # Display question and reference answer
        st.markdown("---")
        st.markdown("### Problem:")
        st.markdown(process_latex(question_data['problem']))
        
        st.markdown("### Reference Answer:")
        st.markdown(process_latex(question_data['reference_answer']))
        
        # Calculate the accuracy of the model
        accuracy = sum(r.get('correct', False) for r in question_data['responses']) / len(question_data['responses'])
        st.markdown("---")
        st.markdown("### Accuracy:")
        st.markdown(f"{accuracy:.2%}")
        
        # Display answer distribution as a list instead of a figure
        st.markdown("---")
        st.markdown("### Answer Distribution:")
        
        # Count model outputs
        model_outputs = [r.get('model_output', '') for r in question_data['responses']]
        counter = Counter(model_outputs)
        
        # Create a DataFrame for better visualization
        df = pd.DataFrame({
            'Answer': list(counter.keys()),
            'Count': list(counter.values())
        }).sort_values('Count', ascending=False)
        
        # Display as a table
        st.table(df)
        
        # Group responses by model output
        model_outputs = {}
        for response in question_data['responses']:
            output = response.get('model_output', '')
            if output not in model_outputs:
                model_outputs[output] = []
            model_outputs[output].append(response)
        
        # Sort model outputs by frequency
        sorted_outputs = sorted(model_outputs.items(), 
                               key=lambda x: len(x[1]), 
                               reverse=True)
        
        # Create a dropdown to select model output
        output_options = [f"{output} ({len(responses)} responses)" 
                         for output, responses in sorted_outputs]
        
        selected_output_idx = st.selectbox(
            "Select model output to view responses",
            range(len(output_options)),
            format_func=lambda i: output_options[i]
        )
        
        if selected_output_idx is not None:
            selected_output, responses = sorted_outputs[selected_output_idx]
            
            # Display a specific response
            st.markdown("---")
            st.markdown(f"### Responses with output: {selected_output}")
            
            # Initialize session state for current response index
            if 'response_index' not in st.session_state:
                st.session_state.response_index = 0
            
            # Ensure response index is valid
            if st.session_state.response_index >= len(responses):
                st.session_state.response_index = 0
            
            # Add navigation buttons
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("Previous") and st.session_state.response_index > 0:
                    st.session_state.response_index -= 1
                    st.rerun()
            with col2:
                if st.button("Next") and st.session_state.response_index < len(responses) - 1:
                    st.session_state.response_index += 1
                    st.rerun()
            
            # Add a selectbox for direct navigation
            response_index = st.selectbox(
                "Select response",
                range(len(responses)),
                key='response_selector',
                index=st.session_state.response_index,
                format_func=lambda x: f"Response {responses[x]['sample_id']} - {'Correct' if responses[x]['correct'] else 'Incorrect'}"
            )
            
            # Update session state when selectbox changes
            st.session_state.response_index = response_index
            
            # Display the selected response
            response = responses[st.session_state.response_index]
            st.markdown(f"#### Sample ID: {response['sample_id']}")
            st.markdown(f"#### Correct: {response['correct']}")
            
            st.markdown("#### Full Response:")
            st.markdown(process_latex(response['full_response']))
            
            # Show raw response in a collapsible section
            with st.expander("Raw Response"):
                st.code(response['full_response'])

# Main Streamlit app
def main(results_file="batch_results/batch_results_chunk_1.jsonl", 
         dataset_name="agentica-org/DeepScaleR-Preview-Dataset"):
    
    st.title("Batch Results Viewer")
    
    data_source = BatchResultsSource(
        results_file=results_file,
        dataset_name=dataset_name
    )
    
    # Add file selector for different chunks
    available_files = [f for f in os.listdir('batch_results') if f.startswith('batch_results_chunk_')]
    if available_files:
        selected_file = st.sidebar.selectbox(
            "Select results file",
            available_files,
            index=available_files.index(os.path.basename(results_file)) if os.path.basename(results_file) in available_files else 0
        )
        results_file = os.path.join('batch_results', selected_file)
        data_source = BatchResultsSource(results_file=results_file, dataset_name=dataset_name)

    try:
        processed_data = data_source.load_data()
        display_data(processed_data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Results Viewer")
    parser.add_argument("--results-file", type=str, 
                        default="batch_results/batch_results_chunk_1.jsonl",
                        help="Path to the batch results file")
    parser.add_argument("--dataset-name", type=str,
                        default="agentica-org/DeepScaleR-Preview-Dataset",
                        help="Name of the dataset to load")
    
    args = parser.parse_args()
    main(results_file=args.results_file, dataset_name=args.dataset_name)