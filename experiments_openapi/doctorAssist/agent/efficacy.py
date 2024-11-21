import os

from .agent import Agent
from .decomposer import DecompositionAgent
from .sub_components.drug_efficacy_detector import EfficacyDetector

# get the path of this file
PATH = os.path.dirname(os.path.abspath(__file__))

config = {
    "model_name": "proteinbert and xgboost",
    "model_path": os.path.join(
        PATH,
        "sub_components/drug_efficacy_detector/models/xgb_DTI.pkl"
        ),
    'model_checkpoint': "facebook/esm2_t6_8M_UR50D",
    "max_length_tokens": 500,
    "device": "cuda",
    "seed": 49
}

tool_list = [
            {
                "type": "function",
                "function": {
                    "name": "retrieval_drugbank",
                    "description": "Given a drug's name, the model retrieves its information from DrugBank database.",
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
            {
                "type": "function",
                "function": {
                    "name": "retrieval_hetionet",
                    "description": '''
                    Given the names of a drug and a disease, the model retrieves the path connecting the drug to the disease from the Hetionet Knowledge Graph. 
                    Hetionet is a comprehensive knowledge graph that integrates diverse biological information by connecting genes, diseases, compounds, and more into an interoperable framework. 
                    It structures real-world biomedical data into a network, facilitating advanced analysis and discovery of new insights into disease mechanisms, drug repurposing, and the genetic underpinnings of health and disease.
                    ''',
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
            {
                "type": "function",
                "function": {
                    "name": "get_efficacy_model_score",
                    "description": '''
                    Given the names of a drug and a disease, the model gives the binding affinity level of the drug and disease.
                    ''',
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
            }
        ]

class EfficacyAgent(Agent):
    def __init__(self, config, depth=1):
        self.agent_tools = tool_list
        self.config = config
        self.depth = depth
        self.detector = EfficacyDetector(config)
        self.reasoning_examples = '''
        
            Question: How can I evaluate the efficacy of the drug aspirin on the disease diabetes?
            Answer:
            <subproblem> Assessing the drug's effectiveness requires retrieving information from the DrugBank database which can be done using "retrieval_drugbank".</subproblem>
            <subproblem> To evaluate the drug's impact on the disease, we must obtain the pathway linking the drug to the disease from the Hetionet Knowledge Graph by using the "retrieval_hetionet" tool. </subproblem>
            <subproblem> To evaluate the binding affinity, we must catculate the binding affinity using "get_efficacy_model_score" tool. </subproblem>
            <subproblem> Offer insights on the drug and disease based on your expertise, without resorting to any external tools. </subproblem>
            
            '''
        self.role = f'''
                    You are an expert in determining efficacy levels of a specific drug for a disease.
                    '''
                    
        super().__init__()
        
        
    def retrieval_drugbank(self, drug_name):
        result = self.detector.retrival_drugbank(drug_name)
        if result == "":
            return None
        else:
            return result

    def retrieval_hetionet(self, drug_name, disease_name):
        result = self.detector.retrival_hetionet(drug_name, disease_name)
        if result == "":
            return None
        else:
            return result
        
    def get_efficacy_model_score(self, drug_name, disease_name):
        result = self.detector.output(drug_name, disease_name)
        if result == "":
            return None
        else:
            return result
        
        
    def process_subproblem(self,subproblem, user_prompt):
        process_prompt = f"The original user problem is: {self.config['base_user_prompt']}\nNow, please you solve this problem: {subproblem}"
        response = self.request(process_prompt)
        
        return response
    
    def combine_agent_results(self, problem_results, prompt):
        return "\n".join(problem_results)
        
    def process(self, drug_name, current_medication):
        current_medication = current_medication.split(",")
        prompt = "Please evaluate the interaction of drug {} with the medication list {}".format(drug_name, current_medication)
        
        decomposer = DecompositionAgent(
            config=self.config,
            tools=self.agent_tools,
            reasoning_examples=self.reasoning_examples
            )

        subproblems = decomposer.process(prompt) 
        
        responses = []
        
        for subproblem in subproblems:
            response = self.process_subproblem(subproblem, prompt)
            responses.append(response)
            
        final_results = self.combine_agent_results(responses, prompt)
            
        return final_results
    