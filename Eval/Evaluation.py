import json
from argparse import ArgumentParser
import re
from vllm import LLM, SamplingParams
import pandas as pd
from datasets import load_dataset
import os

os.makedirs("final_eval", exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."

def final_eval(model_path: str, data_path: str, temperature: float, seed: int) -> None:
    datasets=load_dataset("json",data_files=data_path, split="train")
    model =  LLM(model=model_path, seed=seed, max_model_len=8192)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=4096,
    )

    inputs = []

    for i in range(len(datasets)):
        output = datasets["output"][i]
        answer = datasets["gold"][i]
        Prompt = f"Instruction: Please check whether the generation results is consistent with the gold label.\nGeneration Results:\n{output}\nGold Label:\n{answer}\nPlease output TRUE if they are consistent, otherwise output FALSE."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": Prompt}
        ]
        inputs.append(messages)
    outputs = model.chat(inputs, sampling_params=sampling_params)
    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    for i in range(len(datasets)):
        decoded_output = outputs[i].outputs[0].text
        output_save["input"].append(inputs[i])
        output_save["output"].append(decoded_output)
        output_save["gold"].append(datasets["gold"][i])


        if "TRUE" in decoded_output:
            correct_counts.append(1)
            output_save["correct"].append(1)
        else:
            correct_counts.append(0)
            output_save["correct"].append(0)

    print(f"Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    save_name = data_path.replace("/", "_")
    with open(f"PATH/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="PATH/MODEL")
    parser.add_argument("--data_path", type=str, default="PATH/GENERATED_OUTPUT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    final_eval(args.model_path, args.data_path, args.temperature, args.seed)