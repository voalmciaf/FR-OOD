
import json
from argparse import ArgumentParser
import re
import pandas as pd

# New: use Hugging Face Transformers instead of vLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import random
from MoE import *
from huggingface_hub import login
import tqdm


login(token="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")  # logs in programmatically

SYSTEM_PROMPT = "You are a helpful assistant."
batch_size=16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MERGE_STYLE_TEMPLATE = r"""{{- bos_token }}
{%- set u = none -%}
{%- set a = none -%}
{%- for m in messages -%}
  {%- if m['role'] == 'user' and u is none -%}
    {%- set u = m['content'] | trim -%}
  {%- endif -%}
  {%- if m['role'] == 'assistant' and a is none -%}
    {%- set a = m['content'] | trim -%}
  {%- endif -%}
{%- endfor -%}

### question:
{{- (u or "") }}

### response:
{%- if add_generation_prompt -%}
{%- else -%}
{{- a if a is not none else raise_exception("Training mode requires an assistant answer (add_generation_prompt=False).") -}}
{%- endif -%}
"""

def correct_format(input_text: str) -> str:
    output_text = input_text.replace("\"True\"", "true")
    output_text = output_text.replace("\"False\"", "false")
    return output_text


def evaluate_model_deductive(model_path: str, data_path: str, temperature: float, seed: int, chat: bool) -> None:

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(data_path, "r") as f:
        data = json.load(f)

    # Build inputs list exactly like the original file
    inputs = []
    for item in data:
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["context"]}
            ]
            inputs.append(messages)
        else:
            inputs.append(item["context"])

    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template 
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,

            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
        
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=4096, num_return_sequences=1, do_sample=True,
                                        temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(data[i]["answer"])

            # Keep the original answer extraction logic
            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)

            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                try:
                    predicted_answer_dict = json.loads(predicted_answer)
                    if not isinstance(predicted_answer_dict, dict):
                        correct_counts.append(0)
                        output_save["correct"].append(0)
                        continue
                except json.JSONDecodeError:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
                    continue
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = data[i]["answer"]
            gold_answer = correct_format(gold_answer)
            gold_answer_dict = json.loads(gold_answer)

            all_correct = True
            for key in gold_answer_dict:
                if key in predicted_answer_dict:
                    if predicted_answer_dict[key] != gold_answer_dict[key]:
                        all_correct = False
                        break
                else:
                    all_correct = False
                    break

            if all_correct:
                correct_counts.append(1)
                output_save["correct"].append(1)
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"Deductive_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/deductive", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/deductive/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)

def evaluate_moe_deductive(model_path: str, data_path: str, temperature: float, seed: int, chat: bool) -> None:

    model = LlamaMoEForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(data_path, "r") as f:
        data = json.load(f)

    # Build inputs list exactly like the original file
    inputs = []
    for item in data:
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["context"]}
            ]
            inputs.append(messages)
        else:
            inputs.append(item["context"])

    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template 
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs  
    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=4096, num_return_sequences=1, do_sample=True,
                                        temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(data[i]["answer"])

            # Keep the original answer extraction logic
            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)

            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                try:
                    predicted_answer_dict = json.loads(predicted_answer)
                    if not isinstance(predicted_answer_dict, dict):
                        correct_counts.append(0)
                        output_save["correct"].append(0)
                        continue
                except json.JSONDecodeError:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
                    continue
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = data[i]["answer"]
            gold_answer = correct_format(gold_answer)
            gold_answer_dict = json.loads(gold_answer)

            all_correct = True
            for key in gold_answer_dict:
                if key in predicted_answer_dict:
                    if predicted_answer_dict[key] != gold_answer_dict[key]:
                        all_correct = False
                        break
                else:
                    all_correct = False
                    break

            if all_correct:
                correct_counts.append(1)
                output_save["correct"].append(1)
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"Deductive_MoE_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/deductive", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/deductive/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)

def evaluate_model_inductive(model_path: str, data_path: str, temperature: float, seed: int, chat: bool) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(data_path, "r") as f:
        data = json.load(f)

    inputs = []
    for item in data:
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["context"]}
            ]
            inputs.append(messages)
        else:
            inputs.append(item["context"])

    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template 
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
    
        enc = tokenizer(batch_texts,return_tensors="pt",padding=True,)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens= 4096, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(data[i]["answer"])

            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)

            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                predicted_answer = predicted_answer.strip()
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = data[i]["answer"]

            if predicted_answer == gold_answer:
                correct_counts.append(1)
                output_save["correct"].append(1)
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"Inductive_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/inductive", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/inductive/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)

def evaluate_moe_inductive(model_path: str, data_path: str, temperature: float, seed: int, chat: bool) -> None:
    model = LlamaMoEForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(data_path, "r") as f:
        data = json.load(f)

    inputs = []
    for item in data:
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["context"]}
            ]
            inputs.append(messages)
        else:
            inputs.append(item["context"])

    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template 
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
    
        enc = tokenizer(batch_texts,return_tensors="pt",padding=True,)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens= 4096, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(data[i]["answer"])

            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)

            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                predicted_answer = predicted_answer.strip()
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = data[i]["answer"]

            if predicted_answer == gold_answer:
                correct_counts.append(1)
                output_save["correct"].append(1)
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"Inductive_MoE_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/inductive", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/inductive/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)

def evaluate_model_detective(model_path: str, temperature: float, seed: int, chat: bool) -> None:
    # Evaluate on True Detective Dataset https://github.com/TartuNLP/true-detective
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_path= "PATH/DATA"
    data = pd.read_csv(data_path)

    inputs = []
    gold_answer_list = []
    for i in range(len(data)):
        context = data["mystery_text"][i]
        options = data["answer_options"][i]
        gold_answer = data["answer"][i]
        gold_answer_list.append(gold_answer)

        this_prompt = f"Instruction: Please choose the most possible options based on the information in the story, and response with the answer enclose in <answer><answer>.\nStory: {context}\nAnswer options: {options}"
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": this_prompt}
            ]
            inputs.append(messages)
        else:
            inputs.append(this_prompt)

    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template 
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs
    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
        
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=4096, num_return_sequences=1, do_sample=True,
                                         temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j

            # Keep the original answer extraction logic
            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(gold_answer_list[i])

            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                predicted_answer = predicted_answer.strip()
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = gold_answer_list[i].lower()
            parentheses_match = re.search(r'\((.*?)\)', predicted_answer)
            if parentheses_match:
                predicted_choice = parentheses_match.group(1)
                gold_parentheses_match = re.search(r'\((.*?)\)', gold_answer)
                if gold_parentheses_match:
                    gold_choice = gold_parentheses_match.group(1)

                if predicted_choice == gold_choice:
                    correct_counts.append(1)
                    output_save["correct"].append(1)
                else:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
            else:
                gold_name = gold_answer.split(") ")[1]
                predict_name = predicted_answer.split(") ")[-1]
                if predict_name == gold_name:
                    correct_counts.append(1)
                    output_save["correct"].append(1)
                else:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"Detective_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/detective", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/detective/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)

def evaluate_moe_detective(model_path: str, temperature: float, seed: int, chat: bool) -> None:
    # Evaluate on True Detective Dataset https://github.com/TartuNLP/true-detective
    model = LlamaMoEForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_path = "PATH/DATA"
    data = pd.read_csv(data_path)

    inputs = []
    gold_answer_list = []
    for i in range(len(data)):
        context = data["mystery_text"][i]
        options = data["answer_options"][i]
        gold_answer = data["answer"][i]
        gold_answer_list.append(gold_answer)

        this_prompt = f"Instruction: Please choose the most possible options based on the information in the story, and response with the answer enclose in <answer><answer>.\nStory: {context}\nAnswer options: {options}"
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": this_prompt}
            ]
            inputs.append(messages)
        else:
            inputs.append(this_prompt)

    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template 
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs
    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
        
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=4096, num_return_sequences=1, do_sample=True,
                                        temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j


            # Keep the original answer extraction logic
            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(gold_answer_list[i])

            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                predicted_answer = predicted_answer.strip()
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = gold_answer_list[i].lower()
            parentheses_match = re.search(r'\((.*?)\)', predicted_answer)
            if parentheses_match:
                predicted_choice = parentheses_match.group(1)
                gold_parentheses_match = re.search(r'\((.*?)\)', gold_answer)
                if gold_parentheses_match:
                    gold_choice = gold_parentheses_match.group(1)

                if predicted_choice == gold_choice:
                    correct_counts.append(1)
                    output_save["correct"].append(1)
                else:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
            else:
                gold_name = gold_answer.split(") ")[1]
                predict_name = predicted_answer.split(") ")[-1]
                if predict_name == gold_name:
                    correct_counts.append(1)
                    output_save["correct"].append(1)
                else:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"Detective_MoE_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/detective", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/detective/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)

def evaluate_model_anli(model_path: str, temperature: float, seed: int, chat: bool) -> None:
    # Evaluate on alpha-NLI dataset https://github.com/allenai/abductive-commonsense-reasoning
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    split = "test"
    gold_answer_list = []
    with open(f"PATH/DATA/{split}-labels.lst", "r") as f:
        for line in f.readlines():
            gold_answer_list.append(line.strip())

    data = []
    with open(f"PATH/DATA/{split}.jsonl", "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    inputs = []
    for i in range(len(data)):
        obs1 = data[i]["obs1"]
        obs2 = data[i]["obs2"]
        hyp1 = data[i]["hyp1"]
        hyp2 = data[i]["hyp2"]
        gold_answer = gold_answer_list[i].strip()

        this_prompt = f"Instruction: Based on the beginning and the ending of a story, please choose the option that is more likely than the other, and respond with the answer enclose in <answer><answer>.\nStory beginning: {obs1}\nStory ending: {obs2}\nOptions: (1) {hyp1}\n(2) {hyp2}"
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": this_prompt}
            ]
            inputs.append(messages)
        else:
            inputs.append(this_prompt)
    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template 
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }
    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
        
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=4096, num_return_sequences=1, do_sample=True,
                                        temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(gold_answer_list[i])

            # Keep the original answer extraction logic
            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)

            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                predicted_answer = predicted_answer.strip()
                if predicted_answer == "":
                    correct_counts.append(0)
                    output_save["correct"].append(0)
                    continue
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = gold_answer_list[i]
            if predicted_answer[0] == gold_answer:
                correct_counts.append(1)
                output_save["correct"].append(1)
                continue

            parentheses_match = re.search(r'\((.*?)\)', predicted_answer)
            if parentheses_match:
                predicted_choice = parentheses_match.group(1)

                if predicted_choice == gold_answer:
                    correct_counts.append(1)
                    output_save["correct"].append(1)
                else:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
            else:
                gold_name = data[i]["hyp1"] if gold_answer == "1" else data[i]["hyp2"]
                predict_name = predicted_answer.split(") ")[-1]
                if predict_name.lower() == gold_name.lower():
                    correct_counts.append(1)
                    output_save["correct"].append(1)
                else:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"anli_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/anli", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/anli/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)

def evaluate_moe_anli(model_path: str, temperature: float, seed: int, chat: bool) -> None:
    # Evaluate on alpha-NLI dataset https://github.com/allenai/abductive-commonsense-reasoning
    model = LlamaMoEForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    split="test"
    gold_answer_list = []
    with open(f"PATH/DATA/{split}-labels.lst", "r") as f:
        for line in f.readlines():
            gold_answer_list.append(line.strip())

    data = []
    with open(f"PATH/DATA/{split}.jsonl", "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    inputs = []
    for i in range(len(data)):
        obs1 = data[i]["obs1"]
        obs2 = data[i]["obs2"]
        hyp1 = data[i]["hyp1"]
        hyp2 = data[i]["hyp2"]
        gold_answer = gold_answer_list[i].strip()

        this_prompt = f"Instruction: Based on the beginning and the ending of a story, please choose the option that is more likely than the other, and respond with the answer enclose in <answer><answer>.\nStory beginning: {obs1}\nStory ending: {obs2}\nOptions: (1) {hyp1}\n(2) {hyp2}"
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": this_prompt}
            ]
            inputs.append(messages)
        else:
            inputs.append(this_prompt)
    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template 
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }
    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
        
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=4096, num_return_sequences=1, do_sample=True,
                                        temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(gold_answer_list[i])

            # Keep the original answer extraction logic
            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)


            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                try:
                    predicted_answer = predicted_answer.strip()
                except AttributeError:

                    correct_counts.append(0)
                    output_save["correct"].append(0)
                    continue
                    
                if predicted_answer == "":
                    correct_counts.append(0)
                    output_save["correct"].append(0)
                    continue
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = gold_answer_list[i]
            if predicted_answer[0] == gold_answer:
                correct_counts.append(1)
                output_save["correct"].append(1)
                continue

            parentheses_match = re.search(r'\((.*?)\)', predicted_answer)
            if parentheses_match:
                predicted_choice = parentheses_match.group(1)

                if predicted_choice == gold_answer:
                    correct_counts.append(1)
                    output_save["correct"].append(1)
                else:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
            else:
                gold_name = data[i]["hyp1"] if gold_answer == "1" else data[i]["hyp2"]
                predict_name = predicted_answer.split(") ")[-1]
                if predict_name.lower() == gold_name.lower():
                    correct_counts.append(1)
                    output_save["correct"].append(1)
                else:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"anli_MoE_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/anli", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/anli/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)



def evaluate_model_winowhy(model_path: str, temperature: float, seed: int, chat: bool) -> None:
    # Evaluate on alpha-NLI dataset https://github.com/allenai/abductive-commonsense-reasoning
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = pd.read_csv("PATH/DATA")

    inputs = []
    gold_answer_list = []
    for i in range(len(data)):
        wnli_sent1 = data["wnli_sent1"][i]
        wnli_sent2 = data["wnli_sent2"][i]
        gold_answer = data["label"][i]
        gold_answer_list.append(gold_answer)

        this_prompt = f"Instruction: Based on the given context, is the reasoning shown correct? Please respond with True or False enclosing in <answer><answer>.\nContext: {wnli_sent1}\nReasoning: {wnli_sent2}"
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": this_prompt}
            ]
            inputs.append(messages)
        else:
            inputs.append(this_prompt)
    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template 
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
        
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=4096, num_return_sequences=1, do_sample=True,
                                        temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(gold_answer_list[i])

            # Keep the original answer extraction logic
            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)
            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                if predicted_answer is None:  # 两个 group 都没有匹配到
                    correct_counts.append(0)
                    output_save["correct"].append(0)
                    continue
                predicted_answer = predicted_answer.strip()
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            if gold_answer_list[i] == 1:
                gold_answer = "true"
            else:
                gold_answer = "false"

            if re.search(r'\btrue\b', predicted_answer.lower()):
                predicted_answer_bool = "true"
            elif re.search(r'\bfalse\b', predicted_answer.lower()):
                predicted_answer_bool = "false"
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            if predicted_answer_bool == gold_answer:
                correct_counts.append(1)
                output_save["correct"].append(1)
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"Winowhy_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/winowhy", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/winowhy/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2,default=lambda o: int(o) if isinstance(o, np.int64) else o)

def evaluate_moe_winowhy(model_path: str, temperature: float, seed: int, chat: bool) -> None:
    # Evaluate on alpha-NLI dataset https://github.com/allenai/abductive-commonsense-reasoning
    model = LlamaMoEForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = pd.read_csv("PATH/DATA")

    inputs = []
    gold_answer_list = []
    for i in range(len(data)):
        wnli_sent1 = data["wnli_sent1"][i]
        wnli_sent2 = data["wnli_sent2"][i]
        gold_answer = data["label"][i]
        gold_answer_list.append(gold_answer)

        this_prompt = f"Instruction: Based on the given context, is the reasoning shown correct? Please respond with True or False enclosing in <answer><answer>.\nContext: {wnli_sent1}\nReasoning: {wnli_sent2}"
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": this_prompt}
            ]
            inputs.append(messages)
        else:
            inputs.append(this_prompt)
    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template 
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
        
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=4096, num_return_sequences=1, do_sample=True,
                                        temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(gold_answer_list[i])

            # Keep the original answer extraction logic
            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)
            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                if predicted_answer is None:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
                predicted_answer = predicted_answer.strip()
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            if gold_answer_list[i] == 1:
                gold_answer = "true"
            else:
                gold_answer = "false"

            if re.search(r'\btrue\b', predicted_answer.lower()):
                predicted_answer_bool = "true"
            elif re.search(r'\bfalse\b', predicted_answer.lower()):
                predicted_answer_bool = "false"
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            if predicted_answer_bool == gold_answer:
                correct_counts.append(1)
                output_save["correct"].append(1)
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"Winowhy_MoE_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/winowhy", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/winowhy/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2,default=lambda o: int(o) if isinstance(o, np.int64) else o)


def evaluate_model_folio(model_path: str, temperature: float, seed: int, chat: bool) -> None:
    # Evaluate on FOLIO (V2) validation: yale-nlp/FOLIO

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_dataset(
        "PATH/DATA",
        split="validation"
        ).with_format("pandas")

    inputs = []
    gold_answer_list = []
    for i in range(len(data)):
        premises = data["premises"][i]
        conclusion = data["conclusion"][i]
        gold_answer = data["label"][i]
        gold_answer_list.append(gold_answer)

        this_prompt = f"Instruction: Based on the given premises, is the conclusion correct? Please respond with True, False, or Uncertain enclosing in <answer><answer>.\nPremises: {premises}\nConclusion: {conclusion}"

        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": this_prompt}
            ]
            inputs.append(messages)
        else:
            inputs.append(this_prompt)

    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
        
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=4096, num_return_sequences=1, do_sample=True,
                                        temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            # output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(gold_answer_list[i])


            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                if predicted_answer is None:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
                    continue
                predicted_answer = predicted_answer.strip()
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = gold_answer_list[i].lower()

            if re.search(r'\btrue\b', predicted_answer.lower()):
                predicted_answer_bool = "true"
            elif re.search(r'\bfalse\b', predicted_answer.lower()):
                predicted_answer_bool = "false"
            elif re.search(r'\buncertain\b', predicted_answer.lower()):
                predicted_answer_bool = "uncertain"
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            if predicted_answer_bool == gold_answer:
                correct_counts.append(1)
                output_save["correct"].append(1)
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)

    print(f"Folio Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/folio", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/folio/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)


def evaluate_moe_folio(model_path: str, temperature: float, seed: int, chat: bool) -> None:
    # Evaluate on FOLIO (V2) validation: yale-nlp/FOLIO

    model = LlamaMoEForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_dataset(
        "PATH/DATA",
        split="validation"
        ).with_format("pandas")

    inputs = []
    gold_answer_list = []
    for i in range(len(data)):
        premises = data["premises"][i]
        conclusion = data["conclusion"][i]
        gold_answer = data["label"][i]
        gold_answer_list.append(gold_answer)

        this_prompt = f"Instruction: Based on the given premises, is the conclusion correct? Please respond with True, False, or Uncertain enclosing in <answer><answer>.\nPremises: {premises}\nConclusion: {conclusion}"

        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": this_prompt}
            ]
            inputs.append(messages)
        else:
            inputs.append(this_prompt)

    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
        
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=4096, num_return_sequences=1, do_sample=True,
                                        temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            # output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(gold_answer_list[i])


            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                if predicted_answer is None:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
                    continue
                predicted_answer = predicted_answer.strip()
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = gold_answer_list[i].lower()

            if re.search(r'\btrue\b', predicted_answer.lower()):
                predicted_answer_bool = "true"
            elif re.search(r'\bfalse\b', predicted_answer.lower()):
                predicted_answer_bool = "false"
            elif re.search(r'\buncertain\b', predicted_answer.lower()):
                predicted_answer_bool = "uncertain"
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            if predicted_answer_bool == gold_answer:
                correct_counts.append(1)
                output_save["correct"].append(1)
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)

    print(f"Folio MoE Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/folio", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/folio/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)


def evaluate_model_recv(model_path: str, split: str, temperature: float, seed: int, chat: bool) -> None:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }


    with open(f"PATH/DATA", "r") as f:
        data = json.load(f)

    inputs = []
    gold_answer_list = []
    for i, item in tqdm.tqdm(enumerate(data), total=len(data)):
        context = data[i]["context"]
        question = data[i]["question"]
        gold_answer = "True" if data[i]["answer"]=="A" else "False"
        gold_answer_list.append(gold_answer)

        this_prompt = f"Context: {context}\nInstruction: Based on the given context, please answer the question and enclose the answer in <answer><answer>.\nQuestion: {question}"
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": this_prompt}
            ]
            output_save["input"].append(messages)
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt")
        else:
            output_save["input"].append(this_prompt)
            inputs = tokenizer(this_prompt, return_tensors="pt")

        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=50, top_p=0.95, num_beams=1, max_new_tokens=4096) # some long reasoning cannot finish in 2048

        decoded_output = tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        output_save["output"].append(decoded_output)
        answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)
        output_save["gold"].append(gold_answer_list[i])

        if answer_match:
            predicted_answer = answer_match.group(1) or answer_match.group(2)
            if predicted_answer is None:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue
            predicted_answer = predicted_answer.strip()
        else:
            correct_counts.append(0)
            output_save["correct"].append(0)
            continue

        gold_answer = gold_answer_list[i].lower()

        if re.search(r'\btrue\b', predicted_answer.lower()):
            predicted_answer_bool = "true"
        elif re.search(r'\bfalse\b', predicted_answer.lower()):
            predicted_answer_bool = "false"
        else:
            correct_counts.append(0)
            output_save["correct"].append(0)
            continue

        if predicted_answer_bool == gold_answer:
            correct_counts.append(1)
            output_save["correct"].append(1)
        else:
            correct_counts.append(0)
            output_save["correct"].append(0)

    print(f"RECV_{split} Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    save_name = model_path.replace("/", "_")+f'_recv_{split}'
    save_name = save_name.split(".json")[0]
    os.makedirs("eval_results/recv", exist_ok=True)
    with open(f"eval_results/recv/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)


def evaluate_moe_recv(model_path: str, split: str, temperature: float, seed: int, chat: bool) -> None:
    model = LlamaMoEForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }


    if split == "vitc_moe":
        with open(f"PATH/DATA", "r") as f:
            data = json.load(f)

    if split == "phemeplus_moe":
        with open(f"PATH/DATA", "r") as f:
            data = json.load(f)

    if split == "climate_fever_moe":
        with open(f"PATH/DATA", "r") as f:
            data = json.load(f)

    inputs = []
    gold_answer_list = []
    for i, item in tqdm.tqdm(enumerate(data), total=len(data)):
        context = data[i]["context"]
        question = data[i]["question"]
        gold_answer = "True" if data[i]["answer"]=="A" else "False"
        gold_answer_list.append(gold_answer)

        this_prompt = f"Context: {context}\nInstruction: Based on the given context, please answer the question and enclose the answer in <answer><answer>.\nQuestion: {question}"
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": this_prompt}
            ]
            output_save["input"].append(messages)
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt")
        else:
            output_save["input"].append(this_prompt)
            inputs = tokenizer(this_prompt, return_tensors="pt")

        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=50, top_p=0.95, num_beams=1, max_new_tokens=4096) # some long reasoning cannot finish in 2048

        decoded_output = tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        output_save["output"].append(decoded_output)
        answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)
        output_save["gold"].append(gold_answer_list[i])

        if answer_match:
            predicted_answer = answer_match.group(1) or answer_match.group(2)
            if predicted_answer is None:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue
            predicted_answer = predicted_answer.strip()
        else:
            correct_counts.append(0)
            output_save["correct"].append(0)
            continue

        gold_answer = gold_answer_list[i].lower()

        if re.search(r'\btrue\b', predicted_answer.lower()):
            predicted_answer_bool = "true"
        elif re.search(r'\bfalse\b', predicted_answer.lower()):
            predicted_answer_bool = "false"
        else:
            correct_counts.append(0)
            output_save["correct"].append(0)
            continue

        if predicted_answer_bool == gold_answer:
            correct_counts.append(1)
            output_save["correct"].append(1)
        else:
            correct_counts.append(0)
            output_save["correct"].append(0)

    print(f"RECV_{split} MOE Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    save_name = model_path.replace("/", "_")+f'_recv_{split}'
    save_name = save_name.split(".json")[0]
    os.makedirs("eval_results/recv", exist_ok=True)
    with open(f"eval_results/recv/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)


def evaluate_model_abductive(model_path: str, data_path: str, temperature: float, seed: int, chat: bool) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(data_path, "r") as f:
        data = json.load(f)

    inputs = []
    for item in data:
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["context"]}
            ]
            inputs.append(messages)
        else:
            inputs.append(item["context"])

    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
    
        enc = tokenizer(batch_texts,return_tensors="pt",padding=True,)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens= 4096, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(data[i]["answer"])

            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)

            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                if predicted_answer is None:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
                    continue
                predicted_answer = predicted_answer.strip()
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = data[i]["answer"]

            if predicted_answer == gold_answer:
                correct_counts.append(1)
                output_save["correct"].append(1)
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"Abductive_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/abductive", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/abductive/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)


def evaluate_moe_abductive(model_path: str, data_path: str, temperature: float, seed: int, chat: bool) -> None:
    model = LlamaMoEForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(data_path, "r") as f:
        data = json.load(f)

    inputs = []
    for item in data:
        if chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["context"]}
            ]
            inputs.append(messages)
        else:
            inputs.append(item["context"])

    input_texts = []
    if chat:
        chat_template = tokenizer.chat_template
        for msgs in inputs:
            t = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(t)
    else:
        input_texts = inputs

    correct_counts = []
    output_save = {
        "input": [],
        "output": [],
        "gold": [],
        "correct": []
    }

    n = len(input_texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = input_texts[start:end]
    
        enc = tokenizer(batch_texts,return_tensors="pt",padding=True,)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens= 4096, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=50, top_p=0.95, num_beams=1)
            #output_ids = model.generate(**enc, max_new_tokens=4096,temperature=temperature)
        for j in range(end - start):
            out_ids = output_ids[j]
            gen_only = out_ids[prompt_lens[j]:]
            decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True)
            i = start + j
            output_save["input"].append(inputs[i])
            output_save["output"].append(decoded_output)
            output_save["gold"].append(data[i]["answer"])

            answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', decoded_output, re.DOTALL)

            if answer_match:
                predicted_answer = answer_match.group(1) or answer_match.group(2)
                if predicted_answer is None:
                    correct_counts.append(0)
                    output_save["correct"].append(0)
                    continue
                predicted_answer = predicted_answer.strip()
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
                continue

            gold_answer = data[i]["answer"]

            if predicted_answer == gold_answer:
                correct_counts.append(1)
                output_save["correct"].append(1)
            else:
                correct_counts.append(0)
                output_save["correct"].append(0)
    print("Device Used:",device)

    print(f"Abductive_Moe_Accuracy: {sum(correct_counts) / len(correct_counts) * 100:.2f}%")
    os.makedirs("eval_results/abductive", exist_ok=True)
    save_name = model_path.replace("/", "_")
    with open(f"eval_results/abductive/{save_name}.json", "w") as f:
        json.dump(output_save, f, indent=2)










if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="PATH/MODEL")
    parser.add_argument("--data_path", type=str, default="PATH/DATA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--reasoning_type", type=str, default="deductive")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.reasoning_type == "deductive":
        evaluate_model_deductive(args.model_path, args.data_path, args.temperature, args.seed, args.chat)

    if args.reasoning_type == "deductive_moe":
        evaluate_moe_deductive(args.model_path, args.data_path, args.temperature, args.seed, args.chat)

    if args.reasoning_type == "inductive":
        evaluate_model_inductive(args.model_path, args.data_path, args.temperature, args.seed, args.chat)

    if args.reasoning_type == "inductive_moe":
        evaluate_moe_inductive(args.model_path, args.data_path, args.temperature, args.seed, args.chat)

    if args.reasoning_type == "detective":
        evaluate_model_detective(args.model_path,  args.temperature, args.seed, args.chat)

    if args.reasoning_type == "detective_moe":
        evaluate_moe_detective(args.model_path, args.temperature, args.seed, args.chat)

    if args.reasoning_type == "anli":
        evaluate_model_anli(args.model_path, args.temperature, args.seed, args.chat)

    if args.reasoning_type == "anli_moe":
        evaluate_moe_anli(args.model_path, args.temperature, args.seed, args.chat)

    if args.reasoning_type == "winowhy":
        evaluate_model_winowhy(args.model_path,  args.temperature, args.seed, args.chat)

    if args.reasoning_type == "winowhy_moe":
        evaluate_moe_winowhy(args.model_path,  args.temperature, args.seed, args.chat)

    if args.reasoning_type == "folio":
        evaluate_model_folio(args.model_path,  args.temperature, args.seed, args.chat)

    if args.reasoning_type == "folio_moe":
        evaluate_moe_folio(args.model_path,  args.temperature, args.seed, args.chat)

    if args.reasoning_type == "vitc" or args.reasoning_type == "phemeplus" or args.reasoning_type == "climate_fever":
        evaluate_model_recv(args.model_path, args.reasoning_type, args.temperature, args.seed, args.chat)

    if args.reasoning_type == "vitc_moe" or args.reasoning_type == "phemeplus_moe" or args.reasoning_type == "climate_fever_moe":
        evaluate_moe_recv(args.model_path, args.reasoning_type, args.temperature, args.seed, args.chat)

    if args.reasoning_type == "abductive":
        evaluate_model_abductive(args.model_path, args.data_path, args.temperature, args.seed, args.chat)

    if args.reasoning_type == "abductive_moe":
        evaluate_moe_abductive(args.model_path, args.data_path, args.temperature, args.seed, args.chat)
