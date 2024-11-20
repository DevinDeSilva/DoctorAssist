import random
import numpy as np
import warnings
import torch
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import get_SMILES
warnings.filterwarnings("ignore")


def duckduckgo_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    url = f"https://duckduckgo.com/search?q={query}&num={num_results}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    return _parse_search_results(soup)


def google_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    url = f"https://www.google.com/search?q={query}&num={num_results}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    return _parse_search_results(soup)


def _parse_search_results(soup: BeautifulSoup) -> List[Dict[str, str]]:
    results = []
    for g in soup.find_all("div", class_="g"):
        anchors = g.find_all("a")
        if anchors:
            link = anchors[0]["href"]
            title = g.find("h3")
            title = title.text if title else "No title"
            snippet = g.find("div", class_="VwiC3b")
            snippet = snippet.text if snippet else "No snippet"
            item = {"title": title, "link": link, "snippet": snippet}
            results.append(item)
    return results


def _calculate_individual_score(
    result: Dict[str, str],
    drug_name: str,
    disease_name: str,
    positive_keywords: List[str],
    strong_keywords: List[str],
) -> int:
    score = 0
    text = f"{result['title']} {result['snippet']}".lower()

    if drug_name.lower() in text and disease_name.lower() in text:
        score += 1
    if any(keyword in text for keyword in positive_keywords):
        score += 1
    if any(keyword in text for keyword in strong_keywords):
        score += 1

    return score


class DrugDiseaseInteractionDetector:
    def __init__(self, config):
        self.config = config
        self.__setup__()
        
    def __setup__(self):
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        random.seed(self.config["seed"])
        
        self.search_engine = "google"
        self.num_results = 5
        
            
        
    def search_drug_target(self, text_drug, text_disease):
        query = f"Does {text_drug} have any adverse interaction on {text_disease}"
        if self.search_engine == "google":
            return google_search(query, self.num_results)
        elif self.search_engine == "duckduckgo":
            return duckduckgo_search(query, self.num_results)
        else:
            raise ValueError(
                "Unsupported search engine. Please use 'google' or 'duckduckgo'."
            )
    
    def format_search_results(self, search_results):
        formatted_results = []
        for result in search_results:
            formatted_results.append(f"{result['title']} {result['snippet']}".lower())
        return "\n".join(formatted_results)
        
    
    def calculate_dti_score(self, search_results, text_drug, text_disease)-> float:
        total_score = 0
        max_score = len(search_results) * 3  # 各結果に対して最大3ポイント

        positive_keywords = ["interacts", "binds", "activates", "inhibits", "modulates"]
        strong_keywords = ["strong", "significant", "potent", "effective"]

        for result in search_results:
            score = _calculate_individual_score(
                result, text_drug, text_disease, positive_keywords, strong_keywords
            )
            total_score += score

        if max_score == 0:
            return 0.0

        normalized_score = total_score / max_score
        return round(normalized_score, 2)
        
        
        
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
    
    detector = DrugDiseaseInteractionDetector(config)
    output = detector.output("aspirin", "ibuprofen")