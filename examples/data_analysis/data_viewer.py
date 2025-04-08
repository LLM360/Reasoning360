import streamlit as st
import json
import os
import re
from abc import ABC, abstractmethod
import argparse

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

# Concrete implementation for WandB table JSON files
class VerlWandBTableSource(DataSource):
    def __init__(self, file_path):
        self.file_path = file_path
    
    @property
    def name(self):
        return f"WandB Table: {os.path.basename(self.file_path)}"
    
    def load_data(self):
        with open(self.file_path, 'r') as f:
            raw_data = json.load(f)['data']
        
        processed_data = {}
        for row in raw_data:
            step = row[0]
            for i in range(1, len(row), 3):
                input_text = row[i]
                output = row[i+1]
                correct = row[i+2]
                
                if step not in processed_data:
                    processed_data[step] = []
                processed_data[step].append({
                    'input': input_text,
                    'output': output,
                    'correct': correct
                })
        
        return processed_data

# Helper functions
def exclude_zero_template(prompt):
    return prompt.split("\\boxed{} tag.")[1].split("Assistant:")[0].strip()

def process_latex(text):
    """Convert LaTeX delimiters to ones that Streamlit can render."""
    # Convert \( \) to $ $
    text = text.replace('\\(', '$').replace('\\)', '$')
    # Convert \[ \] to $$ $$
    text = text.replace('\\[', '$$').replace('\\]', '$$')
    
    return text

# Data viewer component
def display_data(processed_data):
    """Display the processed data in the Streamlit UI"""
    if not processed_data:
        st.info("No data available to display.")
        return
    
    # Add a dropdown to select the step
    step = st.selectbox("Select step", list(processed_data.keys()))

    examples = processed_data[step]
    if examples:
        # Initialize session state for current example index
        if 'example_index' not in st.session_state:
            st.session_state.example_index = 0
        
        # Replace radio with selectbox for more compact display
        example_index = st.selectbox(
            "Select example",
            range(len(examples)),
            key='example_selector',
            index=st.session_state.example_index,
            format_func=lambda x: f"Example {x+1} - ({examples[x]['correct']})"
        )
        
        # Update session state when selectbox changes
        st.session_state.example_index = example_index
        
        # Add navigation buttons side by side on the left
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("Previous") and st.session_state.example_index > 0:
                st.session_state.example_index -= 1
                st.rerun()
        with col2:
            if st.button("Next") and st.session_state.example_index < len(examples) - 1:
                st.session_state.example_index += 1
                st.rerun()
        
        # Display the selected example with better formatting
        example = examples[st.session_state.example_index]
        st.subheader(f"Example {st.session_state.example_index + 1}")
        
        st.markdown("---")
        st.markdown("### Correct:")
        st.markdown(example['correct'])
        
        st.markdown("---")
        st.markdown("### Input:")
        st.markdown(process_latex(example['input']))
        st.markdown("#### Raw Input:")
        st.markdown(f"```\n{example['input']}\n```")
        
        st.markdown("---")
        st.markdown("### Output:")
        st.markdown(process_latex(example['output']))
        st.markdown("#### Raw Output:")
        st.markdown(f"```\n{example['output']}\n```")
        st.markdown("---")

# Main Streamlit app
def main(source_type="wandb_table",
         file_path="wandb/run-20250303_183901-v1b5d0cn/files/media/table/val/generations_174_89290fa766917642efa1.table.json"):
    
    st.title("Data Viewer")
    
    if source_type == "wandb_table":
        data_source = VerlWandBTableSource(
            file_path=file_path
        )
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


    processed_data = data_source.load_data()
    display_data(processed_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Viewer for ML experiment outputs")
    parser.add_argument("--source-type", type=str, default="wandb_table", 
                        help="Type of data source (default: wandb_table)")
    parser.add_argument("--file-path", type=str, 
                        default="wandb/run-20250303_183901-v1b5d0cn/files/media/table/val/generations_174_89290fa766917642efa1.table.json",
                        help="Path to the data file")
    
    args = parser.parse_args()
    main(source_type=args.source_type, file_path=args.file_path)