# write a streamlit app to view the data and judge the quality of the data

import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from collections import Counter


# data format
# {"id": {
#     "question_id": int,
#     "question": str,
#     "ground_truth": str,
#     "results": [
#         {
#             "is_correct": [
#                 bool, # whether the answer is correct by math reward function
#                 bool, # whether the answer is correct by the llm judge
#                 str,  # the judge's explanation
#             ],
#             "extracted_output": str, # the extracted output from the llm
#         }
#     ]
# }}

def load_data(file_path):
    """Load data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}

def process_latex(text):
    """Convert LaTeX delimiters to ones that Streamlit can render."""
    if text is None:
        return ""
    # Convert \( \) to $ $
    text = text.replace('\\(', '$').replace('\\)', '$')
    # Convert \[ \] to $$ $$
    text = text.replace('\\[', '$$').replace('\\]', '$$')
    
    return text

def display_data(data):
    """Display the data in the Streamlit UI"""
    if not data:
        st.info("No data available to display.")
        return
    
    # Add a dropdown to select the question
    question_ids = list(data.keys())
    
    # Calculate pass rates for each question
    pass_rates = {}
    for qid in question_ids:
        question_data = data[qid]
        results = question_data.get('results', [])
        if results:
            # Count cases where math reward function marked as correct (index 0 in is_correct)
            math_correct = sum(1 for r in results if r.get('is_correct', [False, False, ""])[0])
            # Count cases where LLM judge marked as correct (index 1 in is_correct)
            llm_judge_correct = sum(1 for r in results if r.get('is_correct', [False, False, ""])[1])
            pass_rates[qid] = {
                "math": f"{math_correct / len(results):.2%}",
                "llm": f"{llm_judge_correct / len(results):.2%}"
            }
        else:
            pass_rates[qid] = {"math": "0.00%", "llm": "0.00%"}
    
    question_id = st.selectbox(
        "Select question ID", 
        question_ids,
        format_func=lambda x: f"Question {x} - Math: {pass_rates[x]['math']} - LLM Judge: {pass_rates[x]['llm']}"
    )
    
    if question_id:
        question_data = data[question_id]
        
        # Display question and ground truth
        st.markdown("---")
        st.markdown("### Question:")
        st.markdown(process_latex(question_data.get('question', '')))
        
        st.markdown("### Ground Truth:")
        st.markdown(process_latex(question_data.get('ground_truth', '')))
        
        # Calculate the accuracy based on LLM judge
        results = question_data.get('results', [])
        if results:
            math_correct = sum(1 for r in results if r.get('is_correct', [False, False, ""])[0])
            llm_judge_correct = sum(1 for r in results if r.get('is_correct', [False, False, ""])[1])
            
            st.markdown("---")
            st.markdown("### Accuracy:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Math Reward Function", f"{math_correct / len(results):.2%}")
            with col2:
                st.metric("LLM Judge", f"{llm_judge_correct / len(results):.2%}")
        
        # Display answer distribution
        st.markdown("---")
        st.markdown("### Answer Distribution:")
        
        # Count extracted outputs and track correctness by both judges
        extracted_outputs = {}
        for r in results:
            output = r.get('extracted_output', '')
            is_correct = r.get('is_correct', [False, False, ""])
            
            if output not in extracted_outputs:
                extracted_outputs[output] = {
                    'count': 0,
                    'math_correct': 0,
                    'llm_correct': 0,
                    'math_incorrect': 0,
                    'llm_incorrect': 0
                }
            
            extracted_outputs[output]['count'] += 1
            if is_correct[0]:  # Math reward function
                extracted_outputs[output]['math_correct'] += 1
            else:
                extracted_outputs[output]['math_incorrect'] += 1
                
            if is_correct[1]:  # LLM judge
                extracted_outputs[output]['llm_correct'] += 1
            elif is_correct[1] == False:
                extracted_outputs[output]['llm_incorrect'] += 1
        
        # Create a DataFrame for better visualization
        df_data = []
        for output, stats in extracted_outputs.items():
            math_accuracy = stats['math_correct'] / stats['count'] if stats['count'] > 0 else 0
            llm_accuracy = stats['llm_correct'] / stats['count'] if stats['count'] > 0 else 0
            
            df_data.append({
                'Answer': output,
                'Count': stats['count'],
                'Math Accuracy': f"{math_accuracy:.2%}",
                'LLM Judge Accuracy': f"{llm_accuracy:.2%}",
                'Math Correct': stats['math_correct'],
                'Math Incorrect': stats['math_incorrect'],
                'LLM Judge Correct': stats['llm_correct'],
                'LLM Judge Incorrect': stats['llm_incorrect']
            })
        
        df = pd.DataFrame(df_data).sort_values('Count', ascending=False)
        
        # Display as a table
        st.table(df)
        
        # Group responses by extracted output
        output_groups = {}
        for response in results:
            output = response.get('extracted_output', '')
            if output not in output_groups:
                output_groups[output] = []
            output_groups[output].append(response)
        
        # Sort outputs by frequency
        sorted_outputs = sorted(output_groups.items(), 
                               key=lambda x: len(x[1]), 
                               reverse=True)
        
        # Create a dropdown to select output
        output_options = [f"{output} ({len(responses)} responses)" 
                         for output, responses in sorted_outputs]
        
        if output_options:
            selected_output_idx = st.selectbox(
                "Select output to view responses",
                range(len(output_options)),
                format_func=lambda i: output_options[i]
            )
            
            if selected_output_idx is not None:
                selected_output, responses = sorted_outputs[selected_output_idx]
                
                # Display responses with this output
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
                    format_func=lambda x: (
                        f"Response {x+1} - " + 
                        ("✅ LLM Judge: Correct" if responses[x].get('is_correct', [False, False, ""])[1] else "❌ LLM Judge: Incorrect")
                    )
                )
                
                # Update session state when selectbox changes
                st.session_state.response_index = response_index
                
                # Display the selected response
                response = responses[st.session_state.response_index]
                is_correct = response.get('is_correct', [False, False, ""])
                
                # Create a colored box based on LLM judge decision
                if is_correct[1]:  # If LLM judge marked as correct
                    st.success("### LLM Judge: Correct ✅")
                else:
                    st.error("### LLM Judge: Incorrect ❌")
                
                # Display math reward function result
                if is_correct[0]:
                    st.info("### Math Reward Function: Correct ✅")
                else:
                    st.warning("### Math Reward Function: Incorrect ❌")
                
                # Display judge's explanation
                st.markdown("### Judge's Explanation:")
                st.markdown(process_latex(is_correct[2]))
                
                # Display extracted output
                st.markdown("### Extracted Output:")
                st.markdown(process_latex(response.get('extracted_output', '')))

def main():
    st.title("LLM Judge Data Viewer")
    
    # Add file selector
    data_dir = "data_analysis"  # Adjust this to your data directory
    if os.path.exists(data_dir):
        available_files = ["grading_results_qwen_32b_instruct.json"]
        if available_files:
            selected_file = st.sidebar.selectbox(
                "Select data file",
                available_files
            )
            file_path = os.path.join(data_dir, selected_file)
            data = load_data(file_path)
            display_data(data)
        else:
            st.sidebar.warning(f"No JSON files found in {data_dir}")
            st.info("Please add JSON files to the data directory.")
    else:
        st.sidebar.warning(f"Data directory '{data_dir}' not found")
        # Allow manual file upload as fallback
        uploaded_file = st.sidebar.file_uploader("Upload a JSON file", type=["json"])
        if uploaded_file is not None:
            data = json.load(uploaded_file)
            display_data(data)

if __name__ == "__main__":
    main()


