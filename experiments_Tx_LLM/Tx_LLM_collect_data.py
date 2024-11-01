import os

os.environ["CUDA_VISIBLE_DEVICES"]="7"

import pickle
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from unsloth import is_bfloat16_supported
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

config  = {
    "dataset":{
        "frac":1,
        },
    "model":{
        "name":"unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "max_seq_len":2048,
        "dtype":None,
        "load_in_4bit":True,
        "peft":{
            "r":32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            "target_modules":["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
            "lora_alpha":16,
            "lora_dropout":0, # Supports any, but = 0 is optimized
            "bias":"none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            "use_gradient_checkpointing":"unsloth", # True or "unsloth" for very long context
            "random_state":3407,
            "use_rslora":False,  # We support rank stabilized LoRA
            "loftq_config":None, # And LoftQ
        },
        
        "training_args":{
            "per_device_train_batch_size":16,
            "gradient_accumulation_steps":4,
            "warmup_steps":5,
            "num_train_epochs":1, # Set this for 1 full training run.
            "learning_rate":1e-4,
            "fp16":not is_bfloat16_supported(),
            "bf16":is_bfloat16_supported(),
            "logging_steps":4,
            "optim":"adamw_8bit",
            "weight_decay":0.01,
            "lr_scheduler_type":"linear",
            "output_dir":"outputs",
        },
        
    },
    "model_save":"saved_model",
    "seed":49,
}

from unsloth import FastLanguageModel
import torch

_, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config["model"]["name"],  # or choose "unsloth/Llama-3.2-1B"
    max_seq_length = config["model"]["max_seq_len"], # Choose any! We auto support RoPE Scaling internally!
    dtype = config["model"]["dtype"],# None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = config["model"]["load_in_4bit"], # Use 4bit quantization to reduce memory usage. Can be False.
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


dataset_paths = [
    "datasets/MTI_miRTarBase/processed",
    "datasets/ADME_BBB_Martins/processed", 
    "datasets/PeptideMHC_MHC1_IEDB-IMGT_Nielsen/processed", 
    "datasets/ADME_HIA_Hou/processed", 
    "datasets/ADME_PAMPA_NCATS/processed", 
    "datasets/ADME_Pgp_Broccatelli/processed", 
    "datasets/Tox_hERG/processed", 
    "datasets/TrialOutcome_phase1/processed", 
    ]

def open_pkl(loc):
    with open(loc, "rb") as f0: 
        data = pickle.load(f0)
        
    return data
        

def load_train_data(path):
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    
    ds_path = os.path.join(path, "train.pkl")
    ds = open_pkl(ds_path)
    ds["text"] = list(map(lambda x: x+EOS_TOKEN,ds["text"]))
    ds = pd.DataFrame.from_dict(ds)
    return ds

def load_test_data(path, _set):
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    
    ds_path = os.path.join(path, f"{_set}.pkl")
    ds = open_pkl(ds_path)
    ds["text"] = list(map(lambda x: x+EOS_TOKEN,ds["text"]))
    ds = pd.DataFrame.from_dict(ds)
    return ds
    

def load_and_prepdata(dataset_paths, ds_set="train", json_folder="final_data", sample=0.5):
    os.makedirs(json_folder,exist_ok=True)
    
    final_df = pd.DataFrame()
    if ds_set == "train":
        for i,p in enumerate(dataset_paths):
            dataset_i = load_train_data(p)
            final_df = pd.concat([final_df, dataset_i], axis=0, ignore_index=True)
  
    else:
        #check if change is required
        for i,p in enumerate(dataset_paths):
            dataset_i = load_test_data(p,ds_set)
            final_df = pd.concat([final_df, dataset_i], axis=0, ignore_index=True)
    
    final_df = final_df.sample(
        frac=sample, 
        random_state=config["seed"]
        )
    
    final_df.to_json(
            os.path.join(json_folder, f"{ds_set}.json"),
            orient="records",
            lines=True
        )
    
    
load_and_prepdata(dataset_paths, sample=config["dataset"]["frac"])
load_and_prepdata(dataset_paths, "valid", sample=config["dataset"]["frac"])
load_and_prepdata(dataset_paths, "test", sample=config["dataset"]["frac"])