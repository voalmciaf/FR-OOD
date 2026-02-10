import json
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import random
import re
from datetime import datetime
from os.path import isfile
from vllm import LLM, SamplingParams


P_LIST = ["history", "finance", "diving", "sport", "writing", "gardening", "shopping", "marketing", "recruiting", "medical", "music"]


SYSTEM_PROMPT = "You are a helpful assistant. You would follow the instructions to answer the question and return the reasoning steps."


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting script at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--dataset", type=str, default="meta-ability-alignment/deductive_data.json")
    parser.add_argument("--sample_rate", type=int, default=5)
    parser.add_argument("--vanilla", action="store_true", default=False)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_prefix = args.model_name.split("/")[-1]
      
    with open(f"{args.dataset}", 'r') as f:
        data_list = json.load(f)

    seed_inputs = [d['context'] for d in data_list]

    data_prefix = args.dataset.replace("/", "_").replace(".json", "")
    out_path = f"trajectories/{model_prefix}_{args.seed}_{data_prefix}.json"

    if isfile(out_path):
        with open(out_path, 'r') as f:
            output_file = json.load(f)
    else:
        output_file = {}

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    llm = LLM(model=args.model_name, seed=args.seed)


    for i in tqdm(range(len(output_file), len(seed_inputs))):
        conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": seed_inputs[i]}
            ]
        conversations = [conversation for _ in range(args.sample_rate)]

        outputs = llm.chat(conversations, sampling_params=sampling_params)

        samples = [output.outputs[0].text for output in outputs]

        question_text = data_list[i]['context']
        gold_answer = data_list[i]["answer"]

        data_id = data_list[i]["id"]
        output_file[data_id] = {
              "question": question_text,
              "gold_answer": gold_answer,
              "samples": samples
            }


        with open(out_path, 'w') as f:
            json.dump(output_file, f, indent=4)
        
    end_time = datetime.now()
    print(f"Finishing script at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_diff = end_time - start_time
    print(f"Time spent: {time_diff}")
