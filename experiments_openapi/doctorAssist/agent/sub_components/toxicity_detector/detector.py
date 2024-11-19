import random
import numpy as np
import os
import warnings
import torch
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..utils.utils import get_SMILES
warnings.filterwarnings("ignore")

class ToxicityDetector:
    def __init__(self, config):
        self.config = config
        self.template_output = "The probability of {} being toxic is {}"
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
        
    def predict(self, text):
        esm2_emb = self.get_esm2_embedding(text)
        xgb_input = esm2_emb.cpu().detach().numpy()
        toxicity = self.xgb_model.predict_proba(xgb_input)
        return toxicity[:,1]
    
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
    
    def output(self, text):
        text_smiles = self.get_smile_string(text)
        toxicity = self.predict(text_smiles)
        
        if isinstance(text, str):
            toxicity = toxicity[0]
            output= self.template_output.format(text, toxicity)
        
        elif isinstance(text, list):
            output = []
            for i, t in enumerate(text):
                output.append(self.template_output.format(t, toxicity[i]))
                
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
        "model_path":"models/xgb_esm2_emb.pkl",
        'model_checkpoint': "facebook/esm2_t6_8M_UR50D",
        "max_length_tokens": 400,
        "device": "cuda",
        "seed": 49
    }
    
    tox_det = ToxicityDetector(config)
    output = tox_det.output("aspirin")
    print(output)