import argparse
import json
import os
import pickle

import torch
from src.utils.io_format import small_model_extract_answer
from src.utils.utils import move_past_key_values_to_device
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


def generate_gt_data(
    small_model, large_model, tokenizer, cot_data, device, device_map, data_type
):
    training_data = []
    analysis_data = []

    for example in tqdm(cot_data, desc="Processing COT Tokens"):
        prompt = example["prompt"]
        cot_text = example["cot"]
        cot_answer = example["answerKey"]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        cot_ids = tokenizer(cot_text, return_tensors="pt").input_ids.to(device)

        prompt_length = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
        small_past_key_values = None
        large_past_key_values = None

        for i in range(len(cot_ids.squeeze())):
            target_token_id = cot_ids[:, i].to(device)
            both_incorrect = False
            route_input_id = None

            with torch.no_grad():
                # small model
                input_ids = input_ids.to(small_model.device)
                if small_past_key_values is not None:
                    small_past_key_values = tuple(
                        [
                            tuple([p.to(small_model.device) for p in layer])
                            for layer in small_past_key_values
                        ]
                    )

                small_outputs = small_model.generate(
                    input_ids=input_ids,
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

                # large model
                input_ids = input_ids.to(large_model.device)
                if large_past_key_values is not None:
                    large_past_key_values = move_past_key_values_to_device(
                        large_past_key_values, device_map
                    )

                large_outputs = large_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    past_key_values=large_past_key_values,
                    use_cache=True,
                )
                large_token_id = large_outputs.sequences[:, -1].unsqueeze(-1).item()
                large_past_key_values = large_outputs.past_key_values

            if small_token_id.item() == target_token_id.item():
                # small model generate the correct token => 1
                label = 1
            elif large_token_id == target_token_id.item():
                # small model get the wrong token, but large's token correct => 0
                label = 0
            else:
                # if both models get the incorrect token:
                both_incorrect = True
                # concat small model generated token id to the input_ids, simulate real inference
                route_input_id = torch.cat(
                    [input_ids.cpu(), small_token_id.cpu()], dim=-1
                )
                # first iteration, only use small model to generate the final response
                label = get_label(
                    small_model, tokenizer, input_ids, cot_answer, data_type
                )

            current_prompt = tokenizer.decode(input_ids.squeeze().tolist())
            target_token_text = tokenizer.decode(target_token_id.squeeze().tolist())
            small_token_text = tokenizer.decode(small_token_id.item())
            large_token_text = tokenizer.decode(large_token_id)

            # get hidden values for the next token (target token)
            input_ids = torch.cat(
                [input_ids.to(device), target_token_id.unsqueeze(0).to(device)], dim=-1
            )

            data_point = {
                "question_id": example["id"],
                "hidden_states": last_hidden_state,
                "label": label,
                "route_input": route_input_id,
                "prompt_length": prompt_length if both_incorrect else None,
                "cot_answer": cot_answer if both_incorrect else None,
            }
            if data_type == "mc" and both_incorrect:
                data_point["choices"] = example["choices"]["text"]
                data_point["labels"] = example["choices"]["label"]
            training_data.append(data_point)

            analysis_data.append(
                {
                    "question_id": example["id"],
                    "current_prompt": current_prompt,
                    "target_token": target_token_text,
                    "small_model_token": small_token_text,
                    "large_model_token": large_token_text,
                    "label": label,
                }
            )

    return training_data, analysis_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth data with small and large models."
    )

    parser.add_argument(
        "--cot_data",
        type=str,
        default="path/to/cot_gt.jsonl",
        help="Path to the CoT data (.jsonl file).",
    )
    parser.add_argument(
        "--large_model", type=str, default="Qwen/Qwen2-72B", help="Large model name."
    )
    parser.add_argument(
        "--small_model_path",
        type=str,
        default="path/to/finetuned-model",
        help="Path to fine-tuned small model or Huggingface repo.",
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default="train_iter_1.pkl",
        help="Output path for the generated training data (.pkl).",
    )
    parser.add_argument(
        "--output_analysis",
        type=str,
        default="analysis_iter_1.pkl",
        help="Output path for the analysis data (.pkl).",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1,2,3",
        help="Specify CUDA devices to use (e.g., '0,1').",
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="Number of process to use."
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank of the current process."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
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

    small_model = AutoModelForCausalLM.from_pretrained(args.small_model_path).to(device)

    large_model = AutoModelForCausalLM.from_pretrained(
        args.large_model, device_map="auto"
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

    training_data, analysis_data = generate_gt_data(
        small_model,
        large_model,
        tokenizer,
        cot_data,
        device,
        device_map,
        args.data_type,
    )

    output_train = args.output_train
    if args.world_size > 1:
        os.makedirs(output_train, exist_ok=True)
        output_train = os.path.join(output_train, f"rank_{args.rank}.pkl")
    else:
        os.makedirs(os.path.dirname(output_train), exist_ok=True)
        output_train = f"{output_train}.pkl"

    with open(output_train, "wb") as f:
        pickle.dump(training_data, f)

    print(f"Training data saved to {output_train}")

    output_analysis = args.output_analysis
    if args.world_size > 1:
        os.makedirs(output_analysis, exist_ok=True)
        output_analysis = os.path.join(output_analysis, f"rank_{args.rank}.pkl")
    else:
        os.makedirs(os.path.dirname(output_analysis), exist_ok=True)
        output_analysis = f"{output_analysis}.pkl"

    with open(output_analysis, "wb") as f:
        pickle.dump(analysis_data, f)

    print(f"Analysis data saved to {output_analysis}")


if __name__ == "__main__":
    main()
