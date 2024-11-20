import random
import numpy as np
import os
import warnings
import torch
import joblib
import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification
cwd_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{cwd_path}/../utils')

from .hetionet import retrieval_hetionet, retrieval_drugbank
from utils import get_SMILES
warnings.filterwarnings("ignore")

class EfficacyDetector:
    def __init__(self, config):
        self.config = config
        self.__setup__()
        
    def __setup__(self):
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        random.seed(self.config["seed"])
        
        if self.config['device'] == 'cpu':
            self.device = "cpu"
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        self.load_esm2_components()
        self.load_xgboost_model()
            
    def load_esm2_components(self):

        model_checkpoint = self.config['model_checkpoint']

        self.emb_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.emb_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(self.device)
        
    def predict(self, text_drug, text_target):
        esm2_emb_drug = self.get_esm2_embedding(text_drug)
        esm2_emb_target = self.get_esm2_embedding(text_target)
        
        xgb_input = np.concatenate([
            esm2_emb_drug.cpu().detach().numpy(), 
            esm2_emb_target.cpu().detach().numpy()], axis=1)
        
        effect_prob = self.xgb_model.predict_proba(xgb_input)
        
        return effect_prob
    
    def get_smile_string(self, drug_name):
        if isinstance(drug_name, str):
            result = get_SMILES(drug_name)
            if result == "":
                return None
            else:
                return result
            
        elif isinstance(drug_name, list):
            smiles = []
            for drug in drug_name:
                result = get_SMILES(drug)
                if result == "":
                    smiles.append(None)
                else:
                    smiles.append(result)
            return smiles
    
    
    def string_formattting(self, drug, target, model_output, n_outs=4):
        str_format = " The a binding score for drug {} for disease {} is {} if it's smaller than 1.5 it is very likely to effect"
        
        str_format = str_format.format(drug, target, model_output)
        return str_format
    
    def retrival_drugbank(self, drug_name):
        result = retrieval_drugbank(drug_name)
        if result == "":
            return None
        else:
            return result
        
    def retrival_hetionet(self, drug_name, disease_name):
        result = retrieval_hetionet(drug_name, disease_name)
        if result == "":
            return None
        else:
            return result
        
    
    
    def output(self, text_drug, text_target):
        
        text_smiles_drug = self.get_smile_string(text_drug)
        text_smiles_target = self.get_smile_string(text_target)
        behavior = self.predict(
            text_smiles_drug, 
            text_smiles_target)
        
        behavior = behavior[0]
        
        output= self.string_formattting(text_drug, text_target, behavior)
                
        return output
    
    def get_esm2_embedding(self, text):
        inputs = self.emb_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.config["max_length_tokens"],
            ).to(self.device)
        
        with torch.no_grad():
            X_drug = self.emb_model.esm.embeddings(**inputs)
            X_drug = self.emb_model.esm.encoder(X_drug)
            X_drug = X_drug.last_hidden_state
            
        return X_drug[:,0,:]
        
    def load_xgboost_model(self):
        self.xgb_model = joblib.load(self.config["model_path"])
        
        
if __name__ == "__main__":
    config = {
        "model_name": "proteinbert and xgboost",
        "model_path_intrinsic": "models/xgb_esm2_emb_intrinsic.pkl",
        "model_path_extrensic": "models/xgb_esm2_emb_extrensic.pkl",
        'model_checkpoint': "facebook/esm2_t6_8M_UR50D",
        "max_length_tokens": 400,
        "device": "cuda",
        "seed": 49
    }
    
    detector = EfficacyDetector(config)
    output = detector.output("aspirin", "ibuprofen")