import os

from .agent import Agent
from .decomposer import DecompositionAgent
from .sub_components.drug_drug_interaction_detector import DrugDrugInteractionDetector

# get the path of this file
PATH = os.path.dirname(os.path.abspath(__file__))

detector_config = {
    "model_name": "proteinbert and xgboost",
    "model_path_intrinsic": os.path.join(
        PATH,
        "sub_components/drug_drug_interaction_detector/models/xgb_DrugBank.pkl"
        ),
    "model_path_extrensic": os.path.join(
        PATH,
        "sub_components/drug_drug_interaction_detector/models/xgb_TWOSIDES.pkl"
        ),
    'model_checkpoint': "facebook/esm2_t6_8M_UR50D",
    "max_length_tokens": 300,
    "device": "cuda",
    "seed": 49
}

tool_list = [
    {
        "type": "function",
        "function": {
            "name": "get_drug_drug_interaction",
            "description": "Given a drug name and the list of Current Medication, return the description of harmfull effects of the drug \
                             with the current medication with probability of them occurring",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "The name of the drug",
                    },
                    
                    "current_medication": {
                        "type": "string",
                        "description": "The names of the medications that the patient is consuming lised in patient profile",
                    }
                },
                "required": ["drug_name", "current_medication"],
            },
        }
    }
]

class DrugMedicationAgent(Agent):
    def __init__(self, config, depth=1):
        self.agent_tools = tool_list
        self.config = config
        self.depth = depth
        self.reasoning_examples = '''
        
            Question: How can I evaluate any adverse interaction of the drug aspirin on the Current Medications [Metformin, Lisinopril, Atorvastatin] in the Patient Profile?
            Answer:
            <subproblem> To evaluate the adverse interaction of the drug aspirin and on current medication we must use the "get_drug_drug_interaction" tool. </subproblem>
            <subproblem> Provide insights into the adverse interaction of the drug on the Current Medication in the Patient Profile, without resorting to any external tools. </subproblem>

        '''
        self.role = f'''
                    You are an expert in determining interaction between two drugs.
                    '''
                    
        super().__init__()
        
    def get_drug_drug_interaction(self, drug_name, current_medication):
        detector = DrugDrugInteractionDetector(detector_config)
        output1 = detector.intrinsic_output(drug_name, current_medication)
        output2 = detector.extrinsic_output(drug_name, current_medication)
        return "{} \n\n {}".format(output1, output2)
    
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
    
        