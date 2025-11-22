from os.path import split

import matplotlib.pyplot as plt
import transformers
from click.core import batch
from datasets import load_dataset, load_from_disk, concatenate_datasets
from functools import partial
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling,set_seed
from transformers import AutoTokenizer
from transformers import optimization
from transformers import AutoModelForCausalLM, TrainingArguments, get_wsd_schedule
import torch
import pandas as pd
import sys
from accelerate import Accelerator
import random
import numpy as np
import re
import os
from accelerate.utils import DistributedDataParallelKwargs
from trl.trainer import DataCollatorForCompletionOnlyLM
from torch.optim import Optimizer
from pathlib import Path
import json
from peft import LoraConfig, LoraModel, get_peft_model, PeftModel
from transformers import TrainerCallback
from datasets import config
from trl import SFTTrainer, SFTConfig

# Set seed for Python
random.seed(42)
# Set seed for NumPy
np.random.seed(42)
# Set seed for PyTorch
torch.manual_seed(42)
if torch.cuda.is_available():
    print("YES")
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
# Optional: Set deterministic behavior for reproducibility (PyTorch)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set seed for HuggingFace Transformers
set_seed(42)

def deductive_text(example):
    example["question"] = example["question"].replace("Below are some formulas connected by conjunctions:", "This is a <Deductive> reasoning task. Below are some formulas connected by conjunctions:")
    example["text"] = f"### question:\n{example['question']}\n\n### response:\n<Deductive>\n{example['trajectory']}"
    return example



def inductive_text(example):
    example["question"] = example["question"].replace("Given the following sequence", "This is a <Inductive> reasoning task. Given the following sequence")
    example["text"] = f"### question:\n{example['question']}\n\n### response:\n<Inductive>\n{example['trajectory']}"
    return example


def abductive_text(example):
    example["question"] = example["question"].replace("Instruction: For each goal, identify which premises directly lead to", "Instruction: This is a <Abductive> reasoning task. For each goal, identify which premises directly lead to")
    example["text"] = f"### question:\n{example['question']}\n\n### response:\n<Abductive>\n{example['trajectory']}"
    return example

# ds_1 = load_dataset("csv",data_files="PATH/DEDUCTIVE_DATA",split='train')
# ds_1 = ds_1.map(deductive_text)
# ds_1 = ds_1.map(lambda x: {"length": len(x["text"].split())})
# ds_1 = ds_1.sort("length")
# ds_1 = ds_1.remove_columns(["question_id","question","gold_answer","trajectory","length"])

ds_2 = load_dataset("csv",data_files="INDUCTIVE_DATA",split='train')
ds_2 = ds_2.map(inductive_text)
ds_2 = ds_2.map(lambda x: {"length": len(x["text"].split())})
ds_2 = ds_2.sort("length")
ds_2 = ds_2.remove_columns(["question_id","question","gold_answer","trajectory","length"])

#
# ds_3 = load_dataset("csv",data_files="ABDUCTIVE_DATA",split='train')
# ds_3 = ds_3.map(abductive_text)
# ds_3 = ds_3.map(lambda x: {"length": len(x["text"].split())})
# ds_3 = ds_3.sort("length")
# ds_3 = ds_3.remove_columns(["question_id","question","gold_answer","trajectory","length"])
# print(ds_3["text"][0])
#
#ds = concatenate_datasets([ds_1, ds_2, ds_3])
ds=ds_2

accelerator = Accelerator()
device = accelerator.device
data_sec=#Depends on the meta-reasoning type
device_num=1
batch_size=2
gradient_accumulation_steps=2
max_step=int(data_sec/(device_num*batch_size*gradient_accumulation_steps))
num_training_steps=max_step

plm_path="PATH/MODEL"
path="PATH/SAVE"
tokenizer = AutoTokenizer.from_pretrained(plm_path)
tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {"additional_special_tokens": ["<answer>", "</answer>", "<think>", "</think>","<Abductive>","<Inductive>","<Deductive>"]}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.save_pretrained(path)

model = AutoModelForCausalLM.from_pretrained(
    plm_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2").to(device)

model.resize_token_embeddings(len(tokenizer))

training_args = SFTConfig(
    output_dir=path,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    logging_steps=1,
    bf16=True,
    max_seq_length=2048,
    max_length=2048,
    dataset_text_field="text",
    save_strategy = "epoch"
)

response_template_with_context = "\n### response:\n"
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
    mlm=False
)

####################Depends on the PEFT method applied, need change here
for param in model.parameters():
    param.requires_grad = False
for i in range(0,model.config.num_hidden_layers):
    trainable=[model.model.layers[i]]
    for j in trainable:
        for param in j.parameters():
            param.requires_grad=True
for param in model.model.embed_tokens.parameters():
    param.requires_grad = True
for param in model.lm_head.parameters():
     param.requires_grad=True

###################################
class CustomTrainer(SFTTrainer):
    def create_optimizer_and_scheduler(self, num_training_steps=max_step):
        """Override to create optimizer and scheduler with custom param groups."""
        self.optimizer = transformers.Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-6,
            weight_decay=0.01,
            scale_parameter=False,
            relative_step=False
        )
        # Create learning rate scheduler
        self.lr_scheduler = get_wsd_schedule(
            self.optimizer,
            num_warmup_steps=int(0.1*num_training_steps),
            num_stable_steps=0,
            num_decay_steps=int(0.9*num_training_steps),
            num_cycles=0.5,
            min_lr_ratio=0.1
        )
###################################
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model(path)

log_history = trainer.state.log_history

with open(path+"/log_history.json", "w") as f:
    json.dump(log_history, f, indent=2)




