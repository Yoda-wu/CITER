import argparse
import json
import os
import time

import torch
from src.utils.io_format import (
    generate_prompt,
    route_extract_answer,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_response(
    model,
    tokenizer,
    questions,
    labels_list,
    choices_list,
    model_device,
    max_new_tokens=500,
):
    prompts = [
        generate_prompt(q, l, c)
        for q, l, c in zip(questions, labels_list, choices_list)
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model_device)

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    torch.cuda.synchronize()
    time_spend = time.perf_counter() - start_time
    decoded = tokenizer.batch_decode(
        outputs.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    scores = []
    for i in range(outputs.sequences.shape[0]):
        score = []
        for j in range(len(outputs.scores)):
            if (
                outputs.sequences[i, j + inputs["input_ids"].shape[1]]
                == tokenizer.eos_token_id
            ):
                break
            s = (
                outputs.scores[j][i]
                .softmax(-1)[outputs.sequences[i, j + inputs["input_ids"].shape[1]]]
                .item()
            )
            score.append(s)
        scores.append(score)

    return time_spend, decoded, scores


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions with either small or large model."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="path/to/test_set.jsonl",
        help="Path to the test dataset in JSONL format.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["large", "small"],
        default="large",
        help="Specify whether to use large or small model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2-72B",
        help="Path to the large model repo or small model fine-tuned path.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-72B",
        help="Pretrained model repository name used to load the tokenizer and config.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="model_results.jsonl",
        help="File to save the inference results.",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1,2,3",
        help="Specify CUDA devices to use (e.g., '0,1').",
    )

    parser.add_argument(
        "--start_index", type=int, default=0, help="Start index for dataset."
    )
    parser.add_argument(
        "--end_index", type=int, default=-1, help="End index for dataset (inclusive)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["mc", "math"],
        default="mc",
        help="Type of data: 'mc' for multiple choice, 'math' for open-ended math problems.",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.model == "large":
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    model_device = next(model.parameters()).device

    dataset = load_jsonl(args.dataset_path)
    print(f"Dataset length: {len(dataset)}")

    if args.end_index == -1:
        args.end_index = len(dataset) - 1

    if args.start_index < 0 or args.start_index >= len(dataset):
        raise ValueError("Invalid start_index, must be within dataset range.")
    if args.end_index < args.start_index or args.end_index >= len(dataset):
        raise ValueError(
            "Invalid end_index, must be within dataset range and >= start_index."
        )

    if args.data_type == "math":
        from src.utils.io_format import generate_math_prompt, extract_math_answer

    print(f"Processing samples from index {args.start_index} to {args.end_index}")

    # run inference
    results = []
    if args.data_type == "mc":
        for batch_start in tqdm(
            range(args.start_index, args.end_index + 1, args.batch_size),
            desc="Processing",
        ):
            batch_end = min(batch_start + args.batch_size - 1, args.end_index)
            batch = dataset[batch_start : batch_end + 1]
            questions = [item["question"] for item in batch]
            choices_list = [item["choices"]["text"] for item in batch]
            labels_list = [item["choices"]["label"] for item in batch]
            true_answers = [item["answerKey"] for item in batch]

            inference_time, responses, scores = get_response(
                model,
                tokenizer,
                questions,
                labels_list,
                choices_list,
                model_device,
            )

            for idx, (item, response, score) in enumerate(
                zip(batch, responses, scores)
            ):
                predict_label, predict_answer = route_extract_answer(
                    response, item["choices"]["text"], item["choices"]["label"]
                )

                results.append(
                    {
                        "id": batch_start + idx,
                        "question": item["question"],
                        "choices": item["choices"],
                        "response": response,
                        "scores": score,
                        "true_answer": item["answerKey"],
                        "predicted_label": predict_label,
                        "inference_time": inference_time / len(batch),
                    }
                )

            current_total = len(results)
            current_avg_time = sum(r["inference_time"] for r in results) / current_total
            print(f"Sofar average inference time: {current_avg_time:.4f} seconds")

    elif args.data_type == "math":
        for i in tqdm(range(args.start_index, args.end_index + 1), desc="Processing"):
            item = dataset[i]
            question = item["problem"]
            true_answer = extract_math_answer(item["solution"])

            prompt = generate_math_prompt(question)
            inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
            prompt_length = inputs.input_ids.shape[1]
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = model.generate(
                **inputs, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id
            )
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            inference_time = end_time - start_time

            generated_ids = outputs[0][prompt_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predict_answer = extract_math_answer(response)

            results.append(
                {
                    "id": i,
                    "problem": question,
                    "true_answer": true_answer,
                    "predicted_label": predict_answer,
                    "inference_time": inference_time,
                }
            )

            print(response)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
