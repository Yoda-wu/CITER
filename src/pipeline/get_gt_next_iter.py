import argparse
import os
import pickle

import torch
from src.utils.io_format import route_extract_answer
from src.models.conf_model import ConfidenceModel
from src.utils.utils import move_past_key_values_to_device
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def route_generate(
    small_model,
    large_model,
    mlp_model,
    tokenizer,
    input_ids,
    prompt_length,
    device_map,
    threshold=0.95,
):
    generated_ids = input_ids[0, prompt_length:].tolist()
    route_small_past_key_values = None
    route_large_past_key_values = None

    token_count = 0

    while True:
        # put input id and all past key values on the same device
        input_ids = input_ids.to(small_model.device)
        if route_small_past_key_values is not None:
            route_small_past_key_values = tuple(
                [
                    tuple([p.to(small_model.device) for p in layer])
                    for layer in route_small_past_key_values
                ]
            )
        # use cache with small past_key_values, only generate 1 new token
        small_outputs = small_model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            past_key_values=route_small_past_key_values,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

        # get new token_id, store past_key_values with the new token data
        token_id = small_outputs.sequences[:, -1].unsqueeze(-1)
        new_small_past_key_values = small_outputs.past_key_values

        last_hidden_state = small_outputs.hidden_states[-1][-1][:, -1, :]

        # pass the mlp head to get confidence score
        with torch.no_grad():
            confidence_score = (
                mlp_model(last_hidden_state.unsqueeze(1)).squeeze().item()
            )

        if confidence_score < threshold:
            if route_large_past_key_values is not None:
                route_large_past_key_values = move_past_key_values_to_device(
                    route_large_past_key_values, device_map
                )
            # use cache with small past_key_values
            large_outputs = large_model.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=1,
                past_key_values=route_large_past_key_values,
                use_cache=True,
                return_dict_in_generate=True,
            )
            # get large model generated new token_id and store past_key_values for large model
            token_id = large_outputs.sequences[:, -1].unsqueeze(-1)
            route_large_past_key_values = large_outputs.past_key_values
        else:
            # if use the small model to generate the new token, update the small_past_key_values with new token data
            route_small_past_key_values = new_small_past_key_values

        # store new generated text and update input_ids
        if token_id.numel() == 1:
            generated_ids.append(token_id.item())
        else:
            generated_ids.extend(token_id.squeeze().tolist())
        input_ids = torch.cat([input_ids, token_id], dim=-1)

        token_count += 1
        # when we use large model, newly generated tokens may be converted into two tokens by small tokenizer
        if token_id.item() == tokenizer.eos_token_id or token_count >= 300:
            break

    final_response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return final_response


# *********** need to be adjusted according to different dataset ***********
def get_label(route_response, answerKey, choices, labels):
    predict_label, predict_answer = route_extract_answer(
        route_response, choices, labels
    )

    # get gt label by comparing predict label with cot answer key
    if predict_label.lower() == answerKey.lower():
        label = 1
    else:
        label = 0

    return label


def generate_gt_data(
    small_model, large_model, tokenizer, gt_data, threshold, mlp_model, device_map
):
    training_data = []

    for example in tqdm(gt_data, desc="Processing COT Tokens"):
        hidden_states = example["hidden_states"]
        route_input = example["route_input"]
        prompt_length = example["prompt_length"]
        cot_answer = example["cot_answer"]
        choices = example["choices"]
        labels = example["labels"]
        label = example["label"]

        if route_input is not None:
            route_response = route_generate(
                small_model=small_model,
                large_model=large_model,
                mlp_model=mlp_model,
                tokenizer=tokenizer,
                input_ids=route_input,
                prompt_length=prompt_length,
                device_map=device_map,
                threshold=threshold,
            )

            # *********** need to be adjusted according to different dataset ***********
            new_label = get_label(
                route_response=route_response,
                answerKey=cot_answer,
                choices=choices,
                labels=labels,
            )

            # move hidden states to cpu
            hidden_states = (
                hidden_states.cpu()
                if isinstance(hidden_states, torch.Tensor)
                else hidden_states
            )

            # *********** need to be adjusted according to different dataset ***********
            training_data.append(
                {
                    "hidden_states": hidden_states,
                    "label": new_label,
                    "route_input": route_input,
                    "prompt_length": prompt_length,
                    "cot_answer": cot_answer,
                    "choices": choices,
                    "labels": labels,
                }
            )

        else:
            hidden_states = (
                hidden_states.cpu()
                if isinstance(hidden_states, torch.Tensor)
                else hidden_states
            )

            # *********** need to be adjusted according to different dataset ***********
            training_data.append(
                {
                    "hidden_states": hidden_states,
                    "label": label,
                    "route_input": None,
                    "prompt_length": None,
                    "cot_answer": None,
                    "choices": None,
                    "labels": None,
                }
            )

    return training_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth data with small and large models."
    )

    parser.add_argument(
        "--gt_data_path",
        type=str,
        default="path/to/train_iter_1.pkl",
        help="Path to the previous iteration output.",
    )
    parser.add_argument(
        "--small_model_path",
        type=str,
        default="path/to/finetuned-model",
        help="Path to fine-tuned small model or Huggingface repo.",
    )
    parser.add_argument(
        "--large_model",
        type=str,
        default="Qwen/Qwen2-72B",
        help="Path to the large model.",
    )
    parser.add_argument(
        "--mlp_model",
        type=str,
        default="/path/to/mlp_model.pth",
        help="Path to the pre-trained MLP model.",
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default="train_iter_2.pkl",
        help="Output path for the updated training data.",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1,2,3",
        help="Specify which CUDA devices to use.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for switching to large model.",
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="Number of process to use."
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank of the current process."
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    with open(args.gt_data_path, "rb") as f:
        gt_data = pickle.load(f)

        if args.world_size > 1:
            chunk_size = len(gt_data) // args.world_size
            remainder = len(gt_data) % args.world_size

            start_idx = args.rank * chunk_size + min(remainder, args.rank)
            end_idx = (args.rank + 1) * chunk_size + min(remainder, args.rank + 1)

            gt_data = gt_data[start_idx:end_idx]

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

    mlp_model_path = args.mlp_model
    mlp_config = AutoConfig.from_pretrained(args.small_model_path)
    mlp_model = ConfidenceModel(mlp_config).to(device)
    mlp_model.load_state_dict(torch.load(mlp_model_path))
    mlp_model.eval()

    device_map = {}
    for name, param in large_model.named_parameters():
        name_parts = name.split(".")
        if len(name_parts) > 2 and name_parts[2].isdigit():
            layer_idx = int(name_parts[2])
            device_map[layer_idx] = str(param.device)

    training_data = generate_gt_data(
        small_model,
        large_model,
        tokenizer,
        gt_data,
        args.threshold,
        mlp_model,
        device_map,
    )

    output_train = args.output_train
    if args.world_size > 1:
        os.makedirs(output_train, exist_ok=True)
        output_train = os.path.join(output_train, f"train_rank_{args.rank}.pkl")
    else:
        os.makedirs(os.path.dirname(output_train), exist_ok=True)
        output_train = f"{output_train}.pkl"

    with open(output_train, "wb") as f:
        pickle.dump(training_data, f)

    print(f"Training data saved to {output_train}")


if __name__ == "__main__":
    main()
