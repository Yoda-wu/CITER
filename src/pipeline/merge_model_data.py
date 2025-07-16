import argparse
import json
import os
import pickle
import torch
import sys
sys.path.append('/home/wyd/CITER/src')
sys.path.append('..')
from utils.io_format import small_model_extract_answer
from tqdm import tqdm

def get_label(small_model, tokenizer, input_ids, cot_answer, data_type):
    """Get label for final answer correctness (only used when both models are incorrect)"""
    # This function is only needed when both models are wrong
    # In our separate approach, we'll skip these cases or handle them differently
    return 0  # Placeholder, not used in merging

def merge_data(intermediate_data, large_model_data, data_type):
    """Merge small model and large model data to create final training data"""
    training_data = []
    analysis_data = []
    
    # Create mapping from question_id to large model tokens
    large_model_map = {item["question_id"]: item["large_model_tokens"] 
                      for item in large_model_data}
    
    for data in tqdm(intermediate_data, desc="Merging data"):
        question_id = data["question_id"]
        prompt = data["prompt"]
        cot_text = data["cot_text"]
        cot_answer = data["cot_answer"]
        prompt_length = data["prompt_length"]
        small_model_tokens = data["small_model_tokens"]
        small_model_hidden_states = data["small_model_hidden_states"]
        cot_token_ids = data["cot_token_ids"]
        choices = data["choices"]
        
        if question_id not in large_model_map:
            print(f"Warning: No large model data for question {question_id}")
            continue
            
        large_model_tokens = large_model_map[question_id]
        
        if len(small_model_tokens) != len(large_model_tokens) or \
           len(small_model_tokens) != len(cot_token_ids):
            print(f"Warning: Token length mismatch for question {question_id}")
            continue
        
        # Reconstruct prompt for each token position
        tokenizer = None  # Will be set in main
        input_ids = None  # Will be reconstructed
        
        for i, (small_token, large_token, hidden_state, target_token) in enumerate(
            zip(small_model_tokens, large_model_tokens, small_model_hidden_states, cot_token_ids)
        ):
            both_incorrect = False
            route_input_id = None
            
            # Determine label based on token matching
            if small_token == target_token:
                # Small model generates correct token => 1
                label = 1
            elif large_token == target_token:
                # Small model wrong, large model correct => 0
                label = 0
            else:
                # Both models incorrect
                both_incorrect = True
                # In this separate approach, we'll skip these cases or handle them differently
                # For now, we'll assign label=0 (prefer large model when both are wrong)
                label = 0
            
            # Create data point
            data_point = {
                "question_id": question_id,
                "hidden_states": hidden_state,
                "label": label,
                "route_input": route_input_id,
                "prompt_length": prompt_length if both_incorrect else None,
                "cot_answer": cot_answer if both_incorrect else None,
            }
            
            if data_type == "mc" and both_incorrect and choices:
                data_point["choices"] = choices["text"] if choices else None
                data_point["labels"] = choices["label"] if choices else None
                
            training_data.append(data_point)
            
            # Create analysis data
            target_token_text = str(target_token)
            small_token_text = str(small_token)
            large_token_text = str(large_token)
            
            analysis_data.append({
                "question_id": question_id,
                "current_prompt": f"{prompt}...",
                "target_token": target_token_text,
                "small_model_token": small_token_text,
                "large_model_token": large_token_text,
                "label": label,
                "token_position": i
            })
    
    return training_data, analysis_data

def main():
    parser = argparse.ArgumentParser(
        description="Merge small and large model data to create final training data."
    )

    parser.add_argument(
        "--intermediate_data",
        type=str,
        default="small_model_intermediate.pkl",
        help="Path to the intermediate data (.pkl file).",
    )
    parser.add_argument(
        "--large_model_data",
        type=str,
        default="large_model_data.pkl",
        help="Path to the large model data (.pkl file).",
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default="train_iter_1.pkl",
        help="Output path for the final training data (.pkl).",
    )
    parser.add_argument(
        "--output_analysis",
        type=str,
        default="analysis_iter_1.pkl",
        help="Output path for the analysis data (.pkl).",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["mc", "math"],
        default="mc",
        help="Type of task: 'mc' for multiple choice, 'math' for open-form math answers.",
    )

    args = parser.parse_args()

    # Load data
    print("Loading intermediate data...")
    with open(args.intermediate_data, "rb") as f:
        intermediate_data = pickle.load(f)
    
    print("Loading large model data...")
    with open(args.large_model_data, "rb") as f:
        large_model_data = pickle.load(f)

    # Merge data
    print("Merging data...")
    training_data, analysis_data = merge_data(
        intermediate_data, large_model_data, args.data_type
    )

    # Save final training data
    output_train = args.output_train
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    output_train = f"{output_train}.pkl"
    
    with open(output_train, "wb") as f:
        pickle.dump(training_data, f)
    print(f"Training data saved to {output_train}")

    # Save analysis data
    output_analysis = args.output_analysis
    os.makedirs(os.path.dirname(output_analysis), exist_ok=True)
    output_analysis = f"{output_analysis}.pkl"
    
    with open(output_analysis, "wb") as f:
        pickle.dump(analysis_data, f)
    print(f"Analysis data saved to {output_analysis}")
    
    # Print summary statistics
    if training_data:
        labels = [item["label"] for item in training_data]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\nTraining data summary:")
        print(f"Total data points: {len(training_data)}")
        for label, count in sorted(label_counts.items()):
            print(f"Label {label}: {count} ({count/len(training_data)*100:.1f}%)")

if __name__ == "__main__":
    main()