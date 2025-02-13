
default_prompt = """
The patient is experiencing osteoarthritis will the use of Ibuprofen be effictive against him and not harmful 
to the patient: 

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
    "device": "cuda:3",
    }


import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def add_current_directory_to_path():
    current_directory = os.getcwd()
    if current_directory not in sys.path:
        sys.path.append(current_directory)
        print(f"Added {current_directory} to Python path.")
    else:
        print(f"{current_directory} is already in the Python path.")
        
add_current_directory_to_path()

from agent.planner_agent import Planner, DecompositionAgent

planner = Planner(config=config)

results = planner.process(default_prompt)
print(results)