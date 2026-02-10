"""
Produce a csv file with training samples with or without using the judge data
"""
import json
import re
import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, default="trajectories/Llama-3.3-70B-Instruct_0_meta-ability-alignment_deductive_data_train.json")
    args = parser.parse_args()

    with open(args.input_path, 'r') as f:
        meta_trajectory_data = json.load(f)

    
    filter_count = 0

    out_dict = {
        "question_id": [],
        "question": [],
        "gold_answer": [],
        "trajectory": [],
    }

    print(f"Total questions in the original file: {len(meta_trajectory_data)}")
    for key, value in meta_trajectory_data.items():
        for i, sample in enumerate(value['samples'][:5]):
            if len(sample.split()) < 20:
                filter_count += 1
                continue
            out_dict["question_id"].append(key)
            out_dict["question"].append(value['question'])

            out_dict["gold_answer"].append(value['gold_answer'])
            out_dict["trajectory"].append(sample)

    print(filter_count)
    out_name = args.input_path.split("/")[-1]
    out_name = out_name.replace(".json", ".csv")
    df = pd.DataFrame(out_dict)
    df.to_csv(out_name, index=False)
    print(f"Total training samples: {len(out_dict['question_id'])}, saved to {out_name}")
    