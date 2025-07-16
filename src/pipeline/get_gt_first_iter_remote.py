
import argparse
import json
import os
import pickle
import torch
import openai

import sys 
sys.path.append('/home/wyd/CITER/src')
sys.path.append('..')
from utils.io_format import small_model_extract_answer
from utils.utils import move_past_key_values_to_device
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['OPENAI_API_KEY'] = 'sk-62ad6f1959274999b2930f7a5382204f'  # 替换为你的 OpenAI API 密钥
os.environ['OPENAI_API_BASE'] = 'https://api.deepseek.com/v1'  # 替换为 DeepSeek API 基础 URL
API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL = os.getenv('OPENAI_API_BASE')
LORA_PATH  ='/home/wyd/CITER/model/checkpoint-18432/'


client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


# <-- 新增：初始化 OpenAI 客户端 -->
# 确保您已设置 OPENAI_API_KEY 环境变量

# <-- 新增：API 调用函数 -->
def get_large_model_api_prediction(prompt_text, tokenizer, api_model_name="deepseek-chat"):
    try:
        response = client.chat.completions.create(
            model=api_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that continues the text."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=1, temperature=0, stop=None
        )
        next_token_text = response.choices[0].message.content
        if not next_token_text:
            return -1 # 返回错误/空值
        next_token_ids = tokenizer.encode(next_token_text, add_special_tokens=False)
        return next_token_ids[0] if next_token_ids else -1
    except Exception as e:
        print(f"调用API时出错: {e}")
        return -1

# get_label 函数保持不变
def get_label(small_model, tokenizer, input_ids, cot_answer, data_type):
    # ... (此函数无需修改)
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


# <-- 修改 generate_gt_data 函数 -->
def generate_gt_data(
    small_model, tokenizer, cot_data, device, api_model_name, data_type # <-- 修改了函数签名
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
        # <-- 移除 large_past_key_values -->

        for i in range(len(cot_ids.squeeze())):
            target_token_id = cot_ids[:, i].to(device)
            both_incorrect = False
            route_input_id = None

            with torch.no_grad():
                # small model 部分保持不变
                input_ids_for_small_model = input_ids.to(small_model.device)
                if small_past_key_values is not None:
                    small_past_key_values = tuple(
                        [
                            tuple([p.to(small_model.device) for p in layer])
                            for layer in small_past_key_values
                        ]
                    )

                small_outputs = small_model.generate(
                    input_ids=input_ids_for_small_model,
                    max_new_tokens=1, pad_token_id=tokenizer.eos_token_id,
                    past_key_values=small_past_key_values, use_cache=True,
                    return_dict_in_generate=True, output_scores=True,
                    output_hidden_states=True,
                )
                small_token_id = small_outputs.sequences[:, -1].unsqueeze(-1)
                small_past_key_values = small_outputs.past_key_values
                last_hidden_state = (
                    small_outputs.hidden_states[-1][-1][:, -1, :].squeeze().cpu()
                )

                # <-- large model 部分被API调用取代 -->
                # 将当前的input_ids解码为文本，发送给API
                current_prompt_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
                large_token_id = get_large_model_api_prediction(
                    current_prompt_text, tokenizer, api_model_name
                )
                
            # ... (后续的label判断和数据记录逻辑基本保持不变) ...
            if small_token_id.item() == target_token_id.item():
                label = 1
            elif large_token_id == target_token_id.item():
                label = 0
            else:
                both_incorrect = True
                route_input_id = torch.cat(
                    [input_ids.cpu(), small_token_id.cpu()], dim=-1
                )
                label = get_label(
                    small_model, tokenizer, input_ids, cot_answer, data_type
                )
            current_prompt = tokenizer.decode(input_ids.squeeze().tolist())
            target_token_text = tokenizer.decode(target_token_id.squeeze().tolist())
            small_token_text = tokenizer.decode(small_token_id.item())
            large_token_text = tokenizer.decode(large_token_id) if large_token_id != -1 else "[API Error]"

            input_ids = torch.cat(
                [input_ids.to(device), target_token_id.unsqueeze(0).to(device)], dim=-1
            )
            data_point = {
                "question_id": example["id"], "hidden_states": last_hidden_state,
                "label": label, "route_input": route_input_id,
                "prompt_length": prompt_length if both_incorrect else None,
                "cot_answer": cot_answer if both_incorrect else None,
            }
            if data_type == "mc" and both_incorrect:
                data_point["choices"] = example["choices"]["text"]
                data_point["labels"] = example["choices"]["label"]
            training_data.append(data_point)
            analysis_data.append(
                {
                    "question_id": example["id"], "current_prompt": current_prompt,
                    "target_token": target_token_text, "small_model_token": small_token_text,
                    "large_model_token": large_token_text, "label": label,
                }
            )

    return training_data, analysis_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth data with a small local model and a large API model."
    )

    # <-- 修改了参数，不再需要 large_model 的本地路径 -->
    parser.add_argument("--cot_data", type=str, default="path/to/cot_gt.jsonl")
    parser.add_argument("--api_model_name", type=str, default="deepseek-chat", help="Large API model name (e.g., gpt-4, gemini-pro).")
    parser.add_argument("--small_model_path", type=str, default="path/to/finetuned-model")
    parser.add_argument("--output_train", type=str, default="train_iter_1.pkl")
    parser.add_argument("--output_analysis", type=str, default="analysis_iter_1.pkl")
    # <-- CUDA设备现在只给小模型用，可以减少数量 -->
    parser.add_argument("--cuda_devices", type=str, default="0", help="Specify CUDA devices to use for the small model.")
    # ... (world_size, rank, batch_size, data_type 等参数保持不变)
    parser.add_argument("--world_size", type=int, default=1, help="Number of process to use.")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the current process.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--data_type", type=str, choices=["mc", "math"], default="mc")


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    with open(args.cot_data, "r") as f:
        # ... (数据分片逻辑保持不变)
        cot_data = [json.loads(line.strip()) for line in f]
        if args.world_size > 1:
            chunk_size = len(cot_data) // args.world_size
            remainder = len(cot_data) % args.world_size
            start_idx = args.rank * chunk_size + min(remainder, args.rank)
            end_idx = (args.rank + 1) * chunk_size + min(remainder, args.rank + 1)
            cot_data = cot_data[start_idx:end_idx]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    small_model = AutoModelForCausalLM.from_pretrained(args.small_model_path).to(device)
    # small_model  = PeftModel.from_pretrained(base_model, LORA_PATH)

    # <-- 移除大模型的加载 -->
    # large_model = AutoModelForCausalLM.from_pretrained(...)

    # <-- Tokenizer 最好用和小模型匹配的，或者一个通用的，这里我们假设用小模型的 -->
    tokenizer = AutoTokenizer.from_pretrained(args.small_model_path)
    tokenizer.padding_side = "left"

    # <-- 移除 device_map 的构建 -->

    # <-- 修改调用generate_gt_data的参数 -->
    training_data, analysis_data = generate_gt_data(
        small_model,
        tokenizer,
        cot_data,
        device,
        args.api_model_name, # 传入API模型名称
        args.data_type,
    )

    # ... (后续保存文件的逻辑保持不变) ...
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