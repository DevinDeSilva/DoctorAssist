import os
import pickle
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from unsloth import is_bfloat16_supported
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

config  = {
    "dataset":{
        "frac":0.1,
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

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config["model"]["name"],  # or choose "unsloth/Llama-3.2-1B"
    max_seq_length = config["model"]["max_seq_len"], # Choose any! We auto support RoPE Scaling internally!
    dtype = config["model"]["dtype"],# None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = config["model"]["load_in_4bit"], # Use 4bit quantization to reduce memory usage. Can be False.
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = config["model"]["peft"]["r"],
    target_modules = config["model"]["peft"]["target_modules"],
    lora_alpha = config["model"]["peft"]["lora_alpha"],
    lora_dropout = config["model"]["peft"]["lora_dropout"],
    bias = config["model"]["peft"]["bias"],
    use_gradient_checkpointing = config["model"]["peft"]["use_gradient_checkpointing"],
    random_state = config["model"]["peft"]["random_state"],
    use_rslora = config["model"]["peft"]["use_rslora"],
    loftq_config = config["model"]["peft"]["loftq_config"],
)


dataset = load_dataset(
    "json",
    data_files={
        "train":"final_data/train.json",
        "valid":"final_data/valid.json",
        "test":"final_data/test.json",
    }
)

response_template = "### Answer:"
#collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset= dataset["valid"],
    dataset_text_field = "text",
    max_seq_length = config["model"]["max_seq_len"],
    dataset_num_proc = 8,
    #data_collator=collator,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = config["model"]["training_args"]["per_device_train_batch_size"],
        gradient_accumulation_steps = config["model"]["training_args"]["gradient_accumulation_steps"],
        warmup_steps = config["model"]["training_args"]["warmup_steps"],
        num_train_epochs = config["model"]["training_args"]["num_train_epochs"], # Set this for 1 full training run.
        learning_rate = config["model"]["training_args"]["learning_rate"],
        fp16 = config["model"]["training_args"]["fp16"],
        bf16 = config["model"]["training_args"]["bf16"],
        logging_steps = config["model"]["training_args"]["logging_steps"],
        optim = config["model"]["training_args"]["optim"],
        weight_decay = config["model"]["training_args"]["weight_decay"],
        lr_scheduler_type = config["model"]["training_args"]["lr_scheduler_type"],
        seed = config["seed"],
        output_dir = config["model"]["training_args"]["output_dir"],
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained(config["model_save"]) # Local saving
tokenizer.save_pretrained(config["model_save"])
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving