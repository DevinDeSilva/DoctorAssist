import os

from .agent import Agent
from .decomposer import DecompositionAgent
from .sub_components.toxicity_detector import ToxicityDetector

# get the path of this file
PATH = os.path.dirname(os.path.abspath(__file__))

config = {
    "model_name": "proteinbert and xgboost",
    "model_path": os.path.join(
        PATH,
        "sub_components/toxicity_detector/models/xgb_esm2_emb.pkl"
        ),
    'model_checkpoint': "facebook/esm2_t6_8M_UR50D",
    "max_length_tokens": 400,
    "device": "cuda",
    "seed": 49
}

tool_list = [
    {
        "type": "function",
        "function": {
            "name": "get_toxicity_detector_score",
            "description": "Given a drug name, return the probability that the drug is harmful to the human body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "The name of the drug",
                    }
                },
                "required": ["drug_name"],
            },
        }
    }
]

class ToxicityAgent(Agent):
    def __init__(self, config, depth=1):
        self.agent_tools = tool_list
        self.config = config
        self.depth = depth
        self.reasoning_examples = ""
        self.role = f'''
                    You are an expert in determining toxicity levels of a specific drug.
                    '''
                    
        super().__init__()
        
    def get_toxicity_detector_score(self, drug_name):
        tox_det = ToxicityDetector(config)
        output = tox_det.output(drug_name)
        return output
    
    def process_subproblem(self,subproblem, user_prompt):
        process_prompt = f"The original user problem is: {self.config['base_user_prompt']}\nNow, please you solve this problem: {subproblem}"
        response = self.request(process_prompt)
        
        return response
    
    def combine_agent_results(self, problem_results, prompt):
        return "\n".join(problem_results)
        
    def process(self, drug_name):
        prompt = "Please evaluate the toxicity of the drug {}".format(drug_name)
        
        decomposer = DecompositionAgent(
            config=self.config,
            tools=self.agent_tools,
            )

        subproblems = decomposer.process(prompt) 
        
        responses = []
        
        for subproblem in subproblems:
            response = self.process_subproblem(subproblem, prompt)
            responses.append(response)
            
        final_results = self.combine_agent_results(responses, prompt)
            
        return final_results
    
        