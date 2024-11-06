
default_prompt = """
The patient is experiencing osteoarthritis will the use of Ibuprofen be effictive against him: 

Patient Profile:
    Age: 56

    Current Diseases:
        Type 2 Diabetes
        Hypertension
        Chronic Kidney Disease (Stage 2) 
        
    Current Medications:
        Metformin (for diabetes)
        Lisinopril (for hypertension)
        Atorvastatin (for high cholesterol)
"""

config = {
    "agent":{
        "toxicity":{
            "type": "function",
            "function": {
                "name": "toxicity_agent",
                "description": "To understand the safety of the drug, including toxicity and side effects, consult the Safety Agent for safety information. Given drug name, return the safety information of the drug, e.g. drug introduction, toxicity and side effects etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_name": {
                            "type": "string",
                            "description": "The drug name",
                       }
                    },
                    "required": ["drug_name"],
                },
            }
        },
        
        "efficacy":{
            "type": "function",
            "function": {
                "name": "efficacy_agent",
                "description": "To assess the drug's efficacy against the diseases, ask the Efficacy Agent for information regarding the drug's effectiveness on the disease. Given drug name and disease name, return the drug introduction, disease introduction, and the path between drug and disease in the hetionet knowledge graph etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_name": {
                            "type": "string",
                            "description": "The drug name",
                        },
                        "disease_name": {
                            "type": "string",
                            "description": "The disease name",
                        }
                    },
                    "required": ["drug_name", "disease_name"],
                },
            }
        },
        
        
        "drug_disease_agent":{
            "type": "function",
            "function": {
                "name": "drug_disease_agent",
                "description": "To understand the negative effects of the drug with the Current Diseases that is stated in Patient Profile",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_name": {
                            "type": "string",
                            "description": "The drug name",
                        },
                        
                        "current_diseases": {
                            "type": "string",
                            "description": "The current diseases the person has",
                        }
                    },
                    "required": ["drug_name", "current_diseases"],
                },
            }
        },
        
        "drug_medication_agent":{
            "type": "function",
            "function": {
                "name": "drug_medication_agent",
                "description": "To understand the negative effects of the drug with the Current Medications that is stated in Patient Profile",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_name": {
                            "type": "string",
                            "description": "The drug name",
                        },
                        
                        "current_medications": {
                            "type": "string",
                            "description": "The current prescribed medication",
                        }
                    },
                    "required": ["drug_name", "current_medications"],
                },
            }
        },
        
        }
    
    }


import sys
import os


def add_current_directory_to_path():
    current_directory = os.getcwd()
    if current_directory not in sys.path:
        sys.path.append(current_directory)
        print(f"Added {current_directory} to Python path.")
    else:
        print(f"{current_directory} is already in the Python path.")
        
add_current_directory_to_path()

from agent.planner import Planner, DecompositionAgent

decom = DecompositionAgent(
    config=config,
    )

results = decom.process(default_prompt)

planner = Planner(config=config)

results = planner.process(results, default_prompt)