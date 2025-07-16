import argparse
import json
import os
import pickle
import torch
import sys
sys.path.append('/home/wyd/CITER/src')
sys.path.append('..')
from utils.io_format import small_model_extract_answer
from utils.utils import move_past_key_values_to_device
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_label(small_model, tokenizer, input_ids, cot_answer, data_type):
    input_ids = input_ids.to(small_model.device)
    small_full_response = small_model.generate(
        input_ids=input_ids,
        max_new_tokens=1000,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    generated_response = tokenizer.decode(
        small_full_response[0], skip_special_tokens=True
    )
    if data_type == "math":
        from src.utils.io_format import extract_math_answer
        generated_answer = extract_math_answer(generated_response)
    else:
        generated_answer = small_model_extract_answer(generated_response)
    if generated_answer is not None and generated_answer.lower() == cot_answer.lower():
        label = 1
    else:
        label = 0
    
    if generated_answer is None:
        print(f"Response cannot extract answer: {generated_response}")
    
    return label

def generate_small_model_data(
    small_model, tokenizer, cot_data, device, data_type
):
    """Generate intermediate data using only the small model"""
    intermediate_data = []
    
    for example in tqdm(cot_data, desc="Processing COT Tokens with Small Model"):
        prompt = example["prompt"]
        cot_text = example["cot"]
        cot_answer = example["answerKey"]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        cot_ids = tokenizer(cot_text, return_tensors="pt").input_ids.to(device)

        prompt_length = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
        small_past_key_values = None
        
        small_model_tokens = []
        small_model_hidden_states = []
        
        for i in range(len(cot_ids.squeeze())):
            target_token_id = cot_ids[:, i].to(device)
            
            with torch.no_grad():
                input_ids_small = input_ids.to(small_model.device)
                if small_past_key_values is not None:
                    small_past_key_values = tuple(
                        [
                            tuple([p.to(small_model.device) for p in layer])
                            for layer in small_past_key_values
                        ]
                    )

                small_outputs = small_model.generate(
                    input_ids=input_ids_small,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                    past_key_values=small_past_key_values,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                )
                small_token_id = small_outputs.sequences[:, -1].unsqueeze(-1)
                small_past_key_values = small_outputs.past_key_values
                last_hidden_state = (
                    small_outputs.hidden_states[-1][-1][:, -1, :].squeeze().cpu()
                )
                
                small_model_tokens.append(small_token_id.item())
                small_model_hidden_states.append(last_hidden_state)
            
            # Update input_ids for next iteration
            input_ids = torch.cat(
                [input_ids.to(device), target_token_id.unsqueeze(0).to(device)], dim=-1
            )
        
        # Store intermediate data
        intermediate_data.append({
            "question_id": example["id"],
            "prompt": prompt,
            "cot_text": cot_text,
            "cot_answer": cot_answer,
            "prompt_length": prompt_length,
            "small_model_tokens": small_model_tokens,
            "small_model_hidden_states": small_model_hidden_states,
            "cot_token_ids": cot_ids.squeeze().tolist(),
            "choices": example.get("choices", None)
        })
    
    return intermediate_data

def main():
    parser = argparse.ArgumentParser(
        description="Generate intermediate data with small model only."
    )

    parser.add_argument(
        "--cot_data",
        type=str,
        default="path/to/cot_gt.jsonl",
        help="Path to the CoT data (.jsonl file).",
    )
    parser.add_argument(
        "--small_model_path",
        type=str,
        default="path/to/finetuned-model",
        help="Path to fine-tuned small model or Huggingface repo.",
    )
    parser.add_argument(
        "--output_intermediate",
        type=str,
        default="small_model_intermediate.pkl",
        help="Output path for the intermediate data (.pkl).",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0",
        help="Specify CUDA devices to use for the small model.",
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="Number of process to use."
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank of the current process."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["mc", "math"],
        default="mc",
        help="Type of task: 'mc' for multiple choice, 'math' for open-form math answers.",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Load CoT data
    with open(args.cot_data, "r") as f:
        cot_data = [json.loads(line.strip()) for line in f]

        if args.world_size > 1:
            chunk_size = len(cot_data) // args.world_size
            remainder = len(cot_data) % args.world_size

            start_idx = args.rank * chunk_size + min(remainder, args.rank)
            end_idx = (args.rank + 1) * chunk_size + min(remainder, args.rank + 1)

            cot_data = cot_data[start_idx:end_idx]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading small model from: {args.small_model_path}")
    small_model = AutoModelForCausalLM.from_pretrained(args.small_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.small_model_path)
    tokenizer.padding_side = "left"

    intermediate_data = generate_small_model_data(
        small_model, tokenizer, cot_data, device, args.data_type
    )

    output_intermediate = args.output_intermediate
    if args.world_size > 1:
        os.makedirs(output_intermediate, exist_ok=True)
        output_intermediate = os.path.join(output_intermediate, f"rank_{args.rank}.pkl")
    else:
        os.makedirs(os.path.dirname(output_intermediate), exist_ok=True)
        output_intermediate = f"{output_intermediate}.pkl"

    with open(output_intermediate, "wb") as f:
        pickle.dump(intermediate_data, f)

    print(f"Intermediate data saved to {output_intermediate}")

if __name__ == "__main__":
    main()