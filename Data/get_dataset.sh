git clone https://github.com/zhiyuanhubj/Meta-Ability-Alignment.git

mkdir raw_data

python get_raw_data.py

python generate_trajectory_vllm.py --model_name "meta-llama/Llama-3.3-70B-Instruct" --dataset "meta-ability-alignment/inductive_data_train.json"

python get_training_samples.py --input-path "trajectories/Llama-3.3-70B-Instruct_0_meta-ability-alignment_deductive_data_train.json"

mkdir sft_samples

python get_sft_samples.py --input-file "trajectories/Llama-3.3-70B-Instruct_0_meta-ability-alignment_deductive_data_train.csv"

python abductive_new_gold.py