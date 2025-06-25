import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def main():
    parser = argparse.ArgumentParser(
        description="Train a causal language model with Chain of Thought data and DeepSpeed configuration."
    )

    # input and output paths
    parser.add_argument(
        "--data_path",
        type=str,
        default="cot_train_data.json",
        help="Path to the COT data (should be 75% of COT data).",
    )
    parser.add_argument(
        "--deepspeed_path",
        type=str,
        default="ds_config.json",
        help="Path to the DeepSpeed configuration file.",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Specify CUDA devices to use, e.g., '0,1,2,3'.",
    )

    # model configurations
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2-7B", help="LLM model name."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model",
        help="Output directory where the model checkpoints will be saved.",
    )

    # training configurations
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=3, help="Batch size per device."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay."
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Limit the total amount of checkpoints.",
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every X steps."
    )
    parser.add_argument(
        "--no-bf16",
        action="store_false",
        dest="bf16",
        help="Disable 16-bit precision during training (default: True).",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by DeepSpeed for distributed training.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)

    train_dataset = load_dataset("json", data_files=args.data_path, split="train")

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"

    def preprocess_function(examples):
        text = [
            prompt + " " + cot + tokenizer.eos_token
            for prompt, cot in zip(examples["prompt"], examples["cot"])
        ]
        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=1000)

        # Set label = -100 for prompt part
        inputs["labels"] = [
            [-100] * len(input_ids) for input_ids in inputs["input_ids"]
        ]
        for i, (input_ids, cot) in enumerate(zip(inputs["input_ids"], examples["cot"])):
            cot_encoded = tokenizer.encode(
                cot + tokenizer.eos_token, add_special_tokens=False
            )
            cot_length = len(cot_encoded)
            start_index = len(input_ids) - cot_length
            inputs["labels"][i][start_index:] = input_ids[start_index : len(input_ids)]

        return inputs

    tokenized_datasets = train_dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        bf16=args.bf16,
        deepspeed=args.deepspeed_path,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    trainer.train()


if __name__ == "__main__":
    main()
