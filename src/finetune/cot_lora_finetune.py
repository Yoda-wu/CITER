import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
# 修正：移除了未使用的 prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

# 设置WandB为离线模式，避免需要登录
os.environ["WANDB_API_KEY"] = 'bdd2db80b2bf94ef45f06029b702ccb5013134eb'
os.environ['WANDB_MODE'] = 'offline'



def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class SavePeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # kwargs中包含了model和tokenizer
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")

        # 我们只在主进程上执行保存操作
        if state.is_world_process_zero:
            # 定义合并后模型的保存路径
            merged_model_path = os.path.join(args.output_dir, "merged_model")
            print(f"训练结束,正在将LoRA权重合并到基础模型中...")
            print(f"合并后的模型将保存在: {merged_model_path}")

            # 调用peft库的merge_and_unload方法
            # 这会返回一个合并了LoRA权重的基础模型
            merged_model = model.merge_and_unload()

            # 保存合并后的模型和对应的tokenizer
            merged_model.save_pretrained(merged_model_path)
            if tokenizer:
                tokenizer.save_pretrained(merged_model_path)
            
            print("合并后的模型已成功保存。")
        
        return control


def main():
    parser = argparse.ArgumentParser(
        description="Train a causal language model with Chain of Thought data and DeepSpeed configuration using LoRA."
    )

    # --- 参数定义部分保持不变 ---
    parser.add_argument("--data_path", type=str, default="cot_train_data.json", help="Path to the COT data.")
    parser.add_argument("--deepspeed_path", type=str, default="ds_config.json", help="Path to the DeepSpeed configuration file.")
    parser.add_argument("--cuda_devices", type=str, default="0", help="Specify CUDA devices to use, e.g., '0,1,2,3'.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B", help="LLM model name.")
    parser.add_argument("--output_dir", type=str, default="model_lora", help="Output directory for model checkpoints.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for LoRA.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Limit the total amount of checkpoints.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps.")
    parser.add_argument("--no-bf16", action="store_false", dest="bf16", help="Disable bf16 training.")
    # 优化：local_rank通常由启动器自动提供，无需手动设置
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training, passed by launcher.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory."
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # DeepSpeed环境会自动处理设备设置
    # if args.local_rank != -1:
    #     torch.cuda.set_device(args.local_rank)

    train_dataset = load_dataset("json", data_files=args.data_path, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": args.local_rank} if args.local_rank != -1 else "auto", # 自动映射设备
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    
    print_trainable_parameters(model)
    # 关键修正：在开启梯度检查点时，必须禁用use_cache以防止计算图中断
    if args.gradient_checkpointing:
        model.config.use_cache = False
    else:
        model.config.use_cache = True


    def preprocess_function(examples):
        text = [
            prompt + " " + cot + tokenizer.eos_token
            for prompt, cot in zip(examples["prompt"], examples["cot"])
        ]
        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=1000)
        inputs["labels"] = [
            [-100] * len(input_ids) for input_ids in inputs["input_ids"]
        ]
        for i, (input_ids, cot) in enumerate(zip(inputs["input_ids"], examples["cot"])):
            cot_encoded = tokenizer.encode(cot + tokenizer.eos_token, add_special_tokens=False)
            cot_length = len(cot_encoded)
            start_index = len(input_ids) - cot_length
            inputs["labels"][i][start_index:] = input_ids[start_index:]
        return inputs

    # 修正：使用更安全的方式移除不再需要的列
    tokenized_datasets = train_dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'cot'])
    
    # 获取环境变量 'LOCAL_RANK' 来判断是否在分布式环境中
    is_distributed = 'LOCAL_RANK' in os.environ
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=10,
        gradient_checkpointing=args.gradient_checkpointing, # 梯度检查点依然开启
        # 修正：仅在分布式环境（由deepspeed启动）下传递deepspeed配置
        deepspeed=args.deepspeed_path if is_distributed else None,
        report_to="wandb", # 启用wandb报告（离线）
    )
    model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        callbacks=[SavePeftModelCallback()],  # 添加自定义回调以保存合并后的模型
    )

    trainer.train()

if __name__ == "__main__":
    main()