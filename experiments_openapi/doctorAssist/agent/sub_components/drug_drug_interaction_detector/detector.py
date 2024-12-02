import random
import numpy as np
import os
import torch
import joblib
import pickle
import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification
cwd_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{cwd_path}/../utils')

from utils import get_SMILES

import warnings
warnings.filterwarnings("ignore")

class DrugDrugInteractionDetector:
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
        self.load_metadata()
        
    def load_metadata(self):
        path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{path}/processed_data/config.pkl", "rb") as f:
            self.data_config = pickle.load(f)
            
    def load_esm2_components(self):

        model_checkpoint = self.config['model_checkpoint']

        self.emb_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.emb_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(self.device)
        
    def predict(self, text_drug1, text_drug2, model="intrinsic"):
        esm2_emb_drug1 = self.get_esm2_embedding(text_drug1)
        esm2_emb_drug2 = self.get_esm2_embedding(text_drug2)
        
        xgb_input = np.concatenate([
            esm2_emb_drug1.cpu().detach().numpy(), 
            esm2_emb_drug2.cpu().detach().numpy()], axis=1)
        
        if model == "intrinsic":
            effect_prob = self.xgb_intrinsic_model.predict_proba(xgb_input)
        else:
            effect_prob = self.xgb_extrensic_model.predict_proba(xgb_input)
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
        
    def string_formattting_intrensic(self, drug1, drug2, model_output, n_outs=4):
        str_format = " with a probability of {}"
        outputs= []
        
        for i in range(len(model_output)):
            orig_class = self.data_config["relabels"]["DrugBank"]["relabel2ori"][i]
            class_str = self.data_config["label_map"]["DrugBank"][orig_class]
            class_str = class_str.replace("#Drug1", drug1)
            class_str = class_str.replace("#Drug2", drug2)
            
            outputs.append((model_output[i], class_str + str_format.format(model_output[i])))
        
        outputs.sort(reverse=True)
        return [x[1] for x in outputs[:n_outs]]
    
    def string_formattting_extrensic(self, drug1, drug2, model_output, n_outs=4):
        str_format = " with a probability of {} when injesting {} and {} together"
        outputs= []
        
        for i in range(len(model_output)):
            orig_class = self.data_config["relabels"]["TWOSIDES"]["relabel2ori"][i]
            class_str = self.data_config["label_map"]["TWOSIDES"][orig_class]            
            outputs.append((model_output[i], class_str + str_format.format(model_output[i], drug1, drug2)))
        
        outputs.sort(reverse=True)
        return [x[1] for x in outputs[:n_outs]]
        
    
    def intrinsic_output(self, text_drug1, text_drug2):
        template_output = "The effect of {} on  {} is of drug properties stated below: \n\n"
        template_output = template_output.format(text_drug1, text_drug2)
        
        text_smiles_drug1 = self.get_smile_string(text_drug1)
        text_smiles_drug2 = self.get_smile_string(text_drug2)
        behavior = self.predict(
            text_smiles_drug1, 
            text_smiles_drug2, 
            model="intrinsic")
        
        behavior = behavior[0]
        output= self.string_formattting_intrensic(text_drug1, text_drug2, behavior)
                
        return template_output+"\n".join(output)
    
    def extrinsic_output(self, text_drug1, text_drug2):
        template_output = "The side effect of the patient injesting {} on  {} together is stated below: \n\n"
        template_output = template_output.format(text_drug1, text_drug2)
        
        text_smiles_drug1 = self.get_smile_string(text_drug1)
        text_smiles_drug2 = self.get_smile_string(text_drug2)
        behavior = self.predict(
            text_smiles_drug1, 
            text_smiles_drug2, 
            model="extrensic")
        
        behavior = behavior[0]
        output= self.string_formattting_extrensic(text_drug1, text_drug2, behavior)
                
        return template_output+"\n".join(output)
    
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
        self.xgb_intrinsic_model = joblib.load(self.config["model_path_intrinsic"])
        self.xgb_extrensic_model = joblib.load(self.config["model_path_extrensic"])
        
        
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
    
    detector = DrugDrugInteractionDetector(config)
    output = detector.intrinsic_output("aspirin", "ibuprofen")
    output = detector.extrinsic_output("aspirin", "ibuprofen")