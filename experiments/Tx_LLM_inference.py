import os
import pickle
from typing import List
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import pipeline
from tqdm.autonotebook import tqdm
from sklearn import metrics
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore", category=ConvergenceWarning)

config  = {
    "model":{
        "max_seq_len":2048,
        "dtype":None,
        "load_in_4bit":True,
        },
    "model_save":"saved_model",
    "batch_size": 16,
    "seed":49,
}

dataset_response_map = {
    'MTI_miRTarBase':{
        "cls_map" : {
            "(A)":0,
            "(B)":1,
        }
    }
}

for v in dataset_response_map.values(): v["num_cls"] = len(v["cls_map"])

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config["model_save"], # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = config["model"]["max_seq_len"],
    dtype = config["model"]["dtype"],
    load_in_4bit = config["model"]["load_in_4bit"],
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

print("Model Loaded !!!", end="\n\n\n")

dataset = load_dataset(
    "json",
    data_files={
        "valid":"final_data/valid.json",
        "test":"final_data/test.json",
    }
)

class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def unnest_dictionary(d:dict, level:str=None, unnested_dict:dict = {}) -> dict:
    for k, v in d.items():
        if level:
            next_level=f"{level}_{k}"
        else:
            next_level = k
            
        if not isinstance(v,dict):
            #print(level,k)
            unnested_dict[next_level] = v
        else:
            #print(d,level,k,v,2)
            unnest_dictionary(v, next_level, unnested_dict)
            
    return unnested_dict

def evalution_categorical(y_pred:np.array, y:np.array) -> dict:
    """
    Contains the relevant metrics for multilabel Binary Classification

    Args:
        y_pred (np.array):model output of logit layer 
        y (np.array): onehot encoded target 
    """
    
    #macro_avg
    
    y = y.flatten()
    
    met = metrics.classification_report(
        y_pred=y_pred,
        y_true=y,
        output_dict=True
    )
    
    print(metrics.classification_report(
        y_pred=y_pred,
        y_true=y,
    ))
    
    met = unnest_dictionary(met,unnested_dict={})
    return met

def generate_categorical_labels_batched(data, dataset, pipe, metadata) -> List[np.array]:
    def map_cls(x):
        try:
            return metadata["cls_map"][x]
        except KeyError:
            return metadata["num_cls"]
    
    labels = []
    
    for out in tqdm(enumerate(pipe(data["text"], batch_size=config["batch_size"])), total=len(data["text"])):
        #(text, model,[tokenizer)
        labels.append(
            out[0]["generated_text"][-3:]
        )
    
    print(np.unique(labels, return_counts=True))
    labels = list(map(map_cls,labels))
    labels = np.array(labels)
    return labels, np.array(data["outputs"])

def generate_categorical_labels_unbatched(data, dataset, pipe, metadata) -> List[np.array]:
    def map_cls(x):
        try:
            return metadata["cls_map"][x]
        except KeyError:
            return metadata["num_cls"]
    
    labels = []
    
    labels = []
    for i,text in tqdm(enumerate(data["text"]), total=len(data["text"])):
        result = pipe(text)#(text, model,[tokenizer)
        labels.append(
            result[0]["generated_text"][-3:]
        )
    
    print(np.unique(labels, return_counts=True))
    labels = list(map(map_cls,labels))
    labels = np.array(labels)
    return labels, np.array(data["outputs"])

def select_datapoints(data, ds_name):
    _ds = ListDataset(data["text"])
    #_ds = DataLoader(dataset, batch_size=config["batch_size"])
    return data, _ds

def dataset_evaluations(data, model, tokenizer, label_map):
    pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=512, 
                        temperature=0.01)
    
    for ds_name, ds_metadata in label_map.items():
        print(f"_"*30)
        print(f"### {ds_name}")
        _data, _ds = select_datapoints(data,ds_name)
        y_pred, y_true = generate_categorical_labels_batched(_data, _ds, pipe, ds_metadata)
        
        _ = evalution_categorical(y_pred, y_true)
        print(f"_"*30, end="\n\n")
        
        
dataset_evaluations(dataset["test"], model, tokenizer, dataset_response_map)


