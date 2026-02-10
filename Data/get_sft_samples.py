import json
import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, default="trajectories/Llama-3.3-70B-Instruct_0_meta-ability-alignment_deductive_data_train.csv")
    parser.add_argument("--save-path", type=str, default="sft_samples")
    args = parser.parse_args()


    data_list = []
    data_df = pd.read_csv(args.input_file)
    for i in range(len(data_df)):
        data_dict = {
            "question": data_df.iloc[i]["question"],
            "response": data_df.iloc[i]["trajectory"]
        }
        data_list.append(data_dict)

    
    save_name = args.input_file.split("/")[-1].replace(".csv", ".json")
    with open(f"{args.save_path}/{save_name}", "w") as f:
        json.dump(data_list, f, indent=4)
