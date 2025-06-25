import argparse
import json
import os
import time

import torch
from src.models.conf_model import ConfidenceModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def generate_response(
    small_model, large_model, mlp_model, prompts, tokenizer, device_map, threshold=0.5
):
    # Batch encode inputs
    tokenizer.padding_side = "left"
    encoded_inputs = tokenizer(
        prompts, padding=True, truncation=True, return_tensors="pt"
    ).to(small_model.device)
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    batch_size = input_ids.size(0)
    prompt_lengths = [len(input_ids[i]) for i in range(batch_size)]

    # Initialize batch variables
    token_counts = [0] * batch_size
    large_token_counts = [0] * batch_size
    small_past_key_values = None
    responses = [""] * batch_size
    eos_flags = [False] * batch_size

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    while not all(eos_flags):
        # Use small model to generate tokens for the batch
        small_outputs = small_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            past_key_values=small_past_key_values,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

        # Get new token IDs and hidden states
        token_ids = small_outputs.sequences[:, -1].unsqueeze(-1)
        small_past_key_values = small_outputs.past_key_values
        last_hidden_states = small_outputs.hidden_states[-1][-1][:, -1, :]

        # Calculate confidence scores using MLP model
        with torch.no_grad():
            confidence_scores = (
                mlp_model(last_hidden_states.unsqueeze(1))
                .squeeze(-1)
                .squeeze(-1)
                .tolist()
            )

        # Collect samples for large model
        large_batch_indices = [
            idx
            for idx in range(batch_size)
            if not eos_flags[idx] and confidence_scores[idx] < threshold
        ]
        if large_batch_indices:
            # Prepare inputs for large model
            large_input_ids = input_ids[large_batch_indices]
            large_attention_mask = attention_mask[large_batch_indices]

            # Generate tokens with large model
            large_outputs = large_model.generate(
                input_ids=large_input_ids,
                attention_mask=large_attention_mask,
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

            # Update large model results back to batch
            for i, idx in enumerate(large_batch_indices):
                token_ids[idx] = large_outputs.sequences[i, -1].unsqueeze(-1)
                large_token_counts[idx] += 1

        # Update token counts and eos flags for all samples
        for idx in range(batch_size):
            if eos_flags[idx]:
                continue

            token_counts[idx] += 1
            if (
                token_ids[idx].item() == tokenizer.eos_token_id
                or token_counts[idx] >= 500
            ):
                eos_flags[idx] = True

        # Update input_ids and attention_mask for the batch
        input_ids = torch.cat([input_ids, token_ids], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.size(0), 1), device=attention_mask.device),
            ],
            dim=1,
        )

    torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_time

    # Decode final responses
    responses = [
        tokenizer.decode(input_ids[i, prompt_lengths[i] :], skip_special_tokens=True)
        for i in range(batch_size)
    ]

    # Calculate final responses and ratios
    large_token_ratios = [
        large_count / total_count if total_count > 0 else 0
        for large_count, total_count in zip(large_token_counts, token_counts)
    ]

    return responses, inference_time, large_token_ratios


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Route-based generation with small and large models."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="path/to/test_set.jsonl",
        help="Path to the test dataset (.jsonl).",
    )
    parser.add_argument(
        "--small_model",
        type=str,
        default="path/to/finetuned-model",
        help="Path to small model or Huggingface repo.",
    )
    parser.add_argument(
        "--large_model", type=str, default="Qwen/Qwen2-72B", help="Large model name."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2-1.5B",
        help="Tokenizer name.",
    )
    parser.add_argument(
        "--mlp_model",
        type=str,
        default="path/to/mlp_model.pth",
        help="Path to the pre-trained MLP model.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of times to run the test."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for routing decisions.",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1,2,3",
        help="Specify which CUDA devices to use (e.g., '0,1').",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="route_iter1_95.jsonl",
        help="Output file for the results.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Specify the batch size for processing.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["mc", "math"],
        default="mc",
        help="Data type: 'mc' for multiple choice, 'math' for open-ended math.",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Load models and tokenizer
    small_model = AutoModelForCausalLM.from_pretrained(args.small_model).to(device)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    large_model = AutoModelForCausalLM.from_pretrained(
        args.large_model, quantization_config=quantization_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # print current type
    for name, param in large_model.named_parameters():
        print(f"Layer: {name}, Data type: {param.dtype}")
        break

    mlp_config = AutoConfig.from_pretrained(args.small_model)
    mlp_model = ConfidenceModel(mlp_config).to(device)
    mlp_model.load_state_dict(torch.load(args.mlp_model))
    mlp_model.eval()

    # Load dataset
    dataset = load_jsonl(args.dataset_path)
    print(f"Dataset length: {len(dataset)}")

    # Map past key value device for large model
    device_map = {}
    for name, param in large_model.named_parameters():
        name_parts = name.split(".")
        if len(name_parts) > 2 and name_parts[2].isdigit():
            layer_idx = int(name_parts[2])
            device_map[layer_idx] = str(param.device)

    # Run generation
    for run_number in range(1, args.num_runs + 1):
        total = len(dataset)
        results = []
        correct_count = 0
        total_time = 0
        processed_count = 0

        batch_size = args.batch_size

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_data = dataset[batch_start:batch_end]
            if args.data_type == "math":
                from src.utils.io_format import generate_math_prompt, extract_math_answer

                questions = [item["problem"] for item in batch_data]
                true_answers = [
                    extract_math_answer(item["solution"]) for item in batch_data
                ]
                prompts = [generate_math_prompt(q) for q in questions]
            else:
                from src.utils.io_format import generate_prompt, route_extract_answer

                questions = [item["question"] for item in batch_data]
                choices_list = [item["choices"]["text"] for item in batch_data]
                labels_list = [item["choices"]["label"] for item in batch_data]
                true_answers = [item["answerKey"] for item in batch_data]
                prompts = [
                    generate_prompt(q, labels, choices)
                    for q, labels, choices in zip(questions, labels_list, choices_list)
                ]

            # Generate responses for the batch
            responses, inference_time, large_token_ratios = generate_response(
                small_model,
                large_model,
                mlp_model,
                prompts,
                tokenizer,
                device_map,
                args.threshold,
            )

            # Process each result in the batch
            for idx, response in enumerate(responses):
                if args.data_type == "math":
                    predict_label = extract_math_answer(response)
                    is_correct = predict_label == true_answers[idx]
                    correct_count += is_correct

                    results.append(
                        {
                            "id": batch_start + idx,
                            "question": questions[idx],
                            "prompt": prompts[idx],
                            "response": response,
                            "true_answer": true_answers[idx],
                            "predicted_label": predict_label,
                            "inference_time": inference_time / len(prompts),
                            "large_token_ratio": large_token_ratios[idx],
                            "is_correct": is_correct,
                        }
                    )
                else:
                    predict_label, predict_answer = route_extract_answer(
                        response, choices_list[idx], labels_list[idx]
                    )
                    is_correct = predict_label == true_answers[idx]
                    correct_count += is_correct

                    results.append(
                        {
                            "id": batch_start + idx,
                            "question": questions[idx],
                            "prompt": prompts[idx],
                            "response": response,
                            "true_answer": true_answers[idx],
                            "predicted_label": predict_label,
                            "predict_answer": predict_answer,
                            "inference_time": inference_time / len(prompts),
                            "large_token_ratio": large_token_ratios[idx],
                            "is_correct": is_correct,
                        }
                    )

                print(prompts[idx])
                print(response)
                print(f"Predicted answer: {predict_label}")
                print(f"True answer: {true_answers[idx]}")
                print(f"large_token_ratio: {large_token_ratios[idx]}")

            processed_count += len(responses)
            total_time += inference_time
            accuracy_so_far = correct_count / processed_count
            average_time_per_sample = total_time / processed_count

            print(f"Processed so far: {processed_count}/{total}")
            print(f"Accuracy so far: {accuracy_so_far:.2%}")
            print(f"Average time so far: {average_time_per_sample:.4f} seconds")

        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        with open(args.output_file, "w") as f:
            for result in results:
                json.dump(result, f)
                f.write("\n")

        print(f"Results for run {run_number} saved to {args.output_file}")


if __name__ == "__main__":
    main()
