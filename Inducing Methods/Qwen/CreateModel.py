from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3MoeForCausalLM
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import transformers
from datasets import load_dataset, load_from_disk, concatenate_datasets
from functools import partial
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling,set_seed
from transformers import AutoTokenizer
from transformers import optimization
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer,GPTNeoXForCausalLM,get_wsd_schedule
import torch
import pandas as pd
import sys
from accelerate import Accelerator
import random
import numpy as np
import re
import os
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import Optimizer
from pathlib import Path
from peft import LoraConfig, LoraModel, get_peft_model, PeftModel
from transformers import TrainerCallback
from datasets import config
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetCount, nvmlDeviceGetUtilizationRates,nvmlDeviceGetMemoryInfo
from huggingface_hub import login
import copy
from peft import LoraConfig, TaskType


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




model = AutoModelForCausalLM.from_pretrained("PATH/BASE_MODEL") #Load Basemodel
tokenizer = AutoTokenizer.from_pretrained("PATH/BASE_MODEL")

# ######################################Model Expansion##########################3
Expand = AutoModelForCausalLM.from_pretrained("PATH/BASE_MODEL",num_hidden_layers=48,
max_window_layers=48,  layer_types=[
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
  ])

index_mapping = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:4,
    5:5,
    6:6,
    7:7,
    8:8,
    9:6,
    10:7,
    11:8,
    12:9,
    13:10,
    14:11,
    15:12,
    16:13,
    17:14,
    18:15,
    19:16,
    20:17,
    21:15,
    22:16,
    23:17,
    24:18,
    25:19,
    26:20,
    27:21,
    28:22,
    29:23,
    30:24,
    31:25,
    32:26,
    33:24,
    34:25,
    35:26,
    36:27,
    37:28,
    38:29,
    39:30,
    40:31,
    41:32,
    42:33,
    43:34,
    44:35,
    45:33,
    46:34,
    47:35
}

for new_idx, old_idx in index_mapping.items():
    Expand.model.layers[new_idx].load_state_dict(model.model.layers[old_idx].state_dict())
for i in[9,10,11,21,22,23,33,34,35,45,46,47]:#These will be the trainable layers
    nn.init.zeros_(Expand.model.layers[i].self_attn.o_proj.weight)  ####Function preservation
    nn.init.zeros_(Expand.model.layers[i].mlp.down_proj.weight)  ####Function preservation

Expand.save_pretrained("PATH/UP-SCALING")
tokenizer.save_pretrained("PATH/UP-SCALING")
#######################################MOE#############################
config = Qwen3Config.from_pretrained("PATH/BASE_MODEL")
config.num_experts = 2
config.num_experts_per_tok = 2
config.decoder_sparse_step = 2
config.mlp_only_layers = []
config.norm_topk_prob = True
config.output_router_logits = True
config.router_aux_loss_coef = 0.01  # for example
config.moe_intermediate_size = config.intermediate_size
Expand = Qwen3MoeForCausalLM.from_pretrained("PATH/BASE_MODEL",config=config)

for i in[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35]:#These will be the trainable layers
    Expand.model.layers[i].mlp.experts[0].load_state_dict(model.model.layers[i].mlp.state_dict())
    Expand.model.layers[i].mlp.experts[1].load_state_dict(model.model.layers[i].mlp.state_dict())


Expand.save_pretrained("PATH/UPCYCLING")
tokenizer.save_pretrained("PATH/UPCYCLING")

