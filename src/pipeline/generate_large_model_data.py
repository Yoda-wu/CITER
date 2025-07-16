import argparse
import json
import os
import pickle
import torch
import sys
sys.path.append('/home/wyd/CITER/src')
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def generate_large_model_data(
    large_model, tokenizer, intermediate_data
):
    """Generate large model predictions using intermediate data"""
    large_model_data = []
    
    for data in tqdm(intermediate_data, desc="Processing with Large Model"):
        question_id = data["question_id"]
        prompt = data["prompt"]
        cot_token_ids = data["cot_token_ids"]
        
        # Get the model's first device for tensor operations
        first_device = next(large_model.parameters()).device
        
        # Reconstruct the conversation up to each token position
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(first_device)
        cot_ids = torch.tensor(cot_token_ids).unsqueeze(0).to(first_device)
        
        large_model_tokens = []
        
        for i in range(len(cot_token_ids)):
            target_token_id = cot_ids[:, i]
            
            with torch.no_grad():
                # Use the model's current device (handled by device_map)
                large_outputs = large_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                large_token_id = large_outputs.sequences[:, -1].unsqueeze(-1).item()
                
                large_model_tokens.append(large_token_id)
            
            # Update input_ids for next iteration - use the model's device
            input_ids = torch.cat(
                [input_ids, target_token_id.unsqueeze(0)], dim=-1
            )
        
        large_model_data.append({
            "question_id": question_id,
            "large_model_tokens": large_model_tokens
        })
    
    return large_model_data

def main():
    parser = argparse.ArgumentParser(
        description="Generate large model predictions using intermediate data."
    )

    parser.add_argument(
        "--intermediate_data",
        type=str,
        default="small_model_intermediate.pkl",
        help="Path to the intermediate data (.pkl file).",
    )
    parser.add_argument(
        "--large_model", 
        type=str, 
        default="Qwen/Qwen2-72B", 
        help="Large model name."
    )
    parser.add_argument(
        "--output_large",
        type=str,
        default="large_model_data.pkl",
        help="Output path for the large model data (.pkl).",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1,2,3",
        help="Specify CUDA devices to use for the large model.",
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="Number of process to use."
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank of the current process."
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Load intermediate data
    with open(args.intermediate_data, "rb") as f:
        intermediate_data = pickle.load(f)

    if args.world_size > 1:
        chunk_size = len(intermediate_data) // args.world_size
        remainder = len(intermediate_data) % args.world_size

        start_idx = args.rank * chunk_size + min(remainder, args.rank)
        end_idx = (args.rank + 1) * chunk_size + min(remainder, args.rank + 1)

        intermediate_data = intermediate_data[start_idx:end_idx]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading large model from: {args.large_model}")
    large_model = AutoModelForCausalLM.from_pretrained(
        args.large_model, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.large_model)
    tokenizer.padding_side = "left"

    # map past key value for large model
    device_map = {}
    for name, param in large_model.named_parameters():
        name_parts = name.split(".")
        if len(name_parts) > 2 and name_parts[2].isdigit():
            layer_idx = int(name_parts[2])
            device_map[layer_idx] = str(param.device)

    large_model_data = generate_large_model_data(
        large_model, tokenizer, intermediate_data
    )

    output_large = args.output_large
    if args.world_size > 1:
        os.makedirs(output_large, exist_ok=True)
        output_large = os.path.join(output_large, f"rank_{args.rank}.pkl")
    else:
        os.makedirs(os.path.dirname(output_large), exist_ok=True)
        output_large = f"{output_large}.pkl"

    with open(output_large, "wb") as f:
        pickle.dump(large_model_data, f)

    print(f"Large model data saved to {output_large}")

if __name__ == "__main__":
    main()