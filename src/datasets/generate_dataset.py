import argparse
import json
import os
import random
import time

import datasets
import openai
import yaml
from tqdm import tqdm


def load_client(key_path="openai_key.yaml"):
    openai._reset_client()
    key = yaml.safe_load(open(key_path))
    for k, v in key.items():
        setattr(openai, k, v)
    return openai._load_client()


client = load_client()


def generate_cot(question, answer):
    prompt = f"Please write an concise explanation for the following question and the given correct answer but do not metion the correct answer directly. Keep the explanation brief and avoid unnecessary details.\n\n{question}\nAnswer: {answer}"
    response = None
    max_try = 50
    while response is None and max_try > 0:
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            response = (
                chat_completion.choices[0].message.content
                + f" Therefore, the answer is {answer}"
            )
        except Exception as e:
            if "filtered" in str(e):
                print("Filtered")
                break
            else:
                print(e)
                max_try -= 1
                time.sleep(60)

    if response is None:
        print("Failed to generate COT")

    return response


def generate_openbookqa_data(sample, index, train=True):
    prompt = (
        f"Fact: {sample['fact1']}\nQuestion: {sample['question_stem']}\nAnswer Choices: "
        + "\t".join(
            [
                f"{label}.{text}"
                for text, label in zip(
                    sample["choices"]["text"], sample["choices"]["label"]
                )
            ]
        )
    )

    instruction = f"Please answer the following question with the given fact: {prompt}"
    data = {
        "id": sample["id"],
        "prompt": instruction,
        "question": instruction,
        "choices": sample["choices"],
        "answerKey": sample["answerKey"],
        "answer": sample["answerKey"],
    }

    if train:
        answer_text = sample["choices"]["text"][ord(sample["answerKey"]) - ord("A")]
        data["cot"] = generate_cot(
            prompt, f"({sample['answerKey'].upper()}). {answer_text}"
        )
        data["question"] = sample["question_stem"]

    return data


def generate_siqa_data(sample, index, train=True):
    prompt = (
        f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer Choices: "
        + "\t".join([f"{label}.{sample[f'answer{label}']}" for label in "ABC"])
    )
    answer_label = chr(ord(sample["label"]) - ord("1") + ord("A"))

    instruction = (
        f"Please answer the following question with the given context: {prompt}"
    )
    data = {
        "id": str(index),
        "prompt": instruction,
        "question": instruction,
        "answerKey": answer_label,
        "answer": answer_label,
    }

    if train:
        answer_text = sample[f"answer{answer_label}"]
        data["cot"] = generate_cot(prompt, f"({answer_label.upper()}). {answer_text}")
        data["question"] = sample["question"]

    return data


def generate_arc_data(sample, index, train=True):
    prompt = f"Question: {sample['question']}\nAnswer Choices: " + "\t".join(
        [
            f"{label}.{text}"
            for text, label in zip(
                sample["choices"]["text"], sample["choices"]["label"]
            )
        ]
    )

    print(sample)
    answer_index = sample["choices"]["label"].index(sample["answerKey"])
    answer_label = chr(answer_index + ord("A"))

    instruction = f"Please answer the following question: {prompt}"
    data = {
        "id": sample["id"],
        "prompt": instruction,
        "question": instruction,
        "choices": sample["choices"],
        "answerKey": answer_label,
        "answer": answer_label,
    }

    if train:
        answer_text = sample["choices"]["text"][answer_index]
        data["cot"] = generate_cot(prompt, f"({answer_label}). {answer_text}")
        data["question"] = sample["question"]

    return data


def generate_mmlu_data(sample, index, train=True):
    # prompt = f"Question: {sample['question']}\nAnswer Choices: " + "\t".join(
    #     [f"{chr(ord('A')+i)}.{text}" for i, text in enumerate(sample["choices"])]
    # )
    prompt = f"Question: {sample['question']}\nChoices:\n" + "\n".join(
        [f"{chr(ord('A')+i)}.{text}" for i, text in enumerate(sample["choices"])]
    )

    answer_label = chr(ord("A") + sample["answer"])

    # instruction = f"Please answer the following question: {prompt}"
    instruction = prompt
    instruction += (
        "Please format your response in the following way:\n"
        "[Explanation]. Therefore, the answer is answer (label).\n"
        "Ensure the final sentence includes the answer followed by the label in parentheses.\n"
        "Answer and Reasoning:"
    )
    data = {
        "id": str(index),
        "prompt": instruction,
        # "question": prompt,
        "question": sample["question"],
        "choices": {
            "text": sample["choices"],
            "label": [chr(ord("A") + i) for i in range(len(sample["choices"]))],
        },
        "answerKey": answer_label,
        # "answer": answer_label,
    }

    if train:
        answer_text = sample["choices"][sample["answer"]]
        # data["cot"] = generate_cot(prompt, f"({answer_label}). {answer_text}")
        data["cot"] = generate_cot(prompt, f"{answer_text} ({answer_label}).")
        # data["question"] = sample["question"]

    return data


def generate_data(sample, index, dataset_name, train=True):
    if "arc" in dataset_name:
        func = generate_arc_data
    elif "mmlu" in dataset_name:
        func = generate_mmlu_data
    else:
        func_map = {"openbookqa": generate_openbookqa_data, "siqa": generate_siqa_data}
        func = func_map[dataset_name]
    return func(sample, index, train)


def load_dataset(dataset_name):
    if "arc" in dataset_name:
        name = {"easy": "ARC-Easy", "challenge": "ARC-Challenge"}[
            dataset_name.remove_prefix("arc_")
        ]
        dataset = datasets.load_dataset("allenai/ai2_arc", name)
    elif "mmlu" in dataset_name:
        name = dataset_name[dataset_name.find("_") + 1 :]
        dataset = datasets.load_dataset("cais/mmlu", name)
    elif dataset_name == "openbookqa":
        dataset = datasets.load_dataset("allenai/openbookqa", "additional")
    elif dataset_name == "siqa":
        dataset = datasets.load_dataset("allenai/social_i_qa", trust_remote_code=True)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate dataset.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="openbookqa",
        help="dataset to generate, available datasets: openbookqa, siqa, arc_easy, arc_challenge, mmlu_professional_law, mmlu_moral_scenarios",
    )

    args = parser.parse_args()

    output_path = f"data/{args.dataset}"
    dataset = load_dataset(args.dataset)

    data = []
    for i, sample in enumerate(
        tqdm(
            dataset["train" if "mmlu" not in args.dataset else "test"],
            desc="Processing Training Data",
        )
    ):
        sample = generate_data(sample, i, args.dataset)
        if not sample["cot"]:
            continue
        data.append(sample)

    random.shuffle(data)

    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/train_small_model.json", "w") as f:
        for sample in data[len(data) // 4 :]:
            json.dump(sample, f)
            f.write("\n")
    with open(f"{output_path}/train_mlp.json", "w") as f:
        for sample in data[: len(data) // 4]:
            json.dump(sample, f)
            f.write("\n")

    data = []
    for i, sample in enumerate(
        tqdm(dataset["validation"], desc="Processing Validation Data")
    ):
        data.append(generate_data(sample, i, args.dataset, train=False))

    with open(f"{output_path}/test.json", "w") as f:
        for sample in data:
            json.dump(sample, f)
            f.write("\n")


if __name__ == "__main__":
    main()
