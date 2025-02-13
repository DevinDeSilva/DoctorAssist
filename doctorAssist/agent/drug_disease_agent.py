from .agent import Agent
from .decomposer import DecompositionAgent
from .sub_components.drug_disease_interaction_detector import DrugDiseaseInteractionDetector

tool_list = [
    {
            "type": "function",
            "function": {
                "name": "internet_search",
                "description": "Search the internet for information on the drug has any adverse effect on the disease",
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
        }
]

class DrugDiseaseAgent(Agent):
    def __init__(self, config, depth=1):
        self.agent_tools = tool_list
        self.config = config
        self.depth = depth
        self.reasoning_examples = '''
                
            Question: How can I evaluate any adverse effect of the drug aspirin on the Current Diseases [Type 2 Diabetes, Hypertension, Chronic Kidney Disease (Stage 2)] in the Patient Profile?
            Answer:
            <subproblem> To evaluate the adverse effect of the drugs, we must use the "internet_search" tool to check if any drug has ny adverse effect on the diseases. </subproblem>
            <subproblem> Provide insights into the adverse effect of the drug on the Current Diseases in the Patient Profile, without resorting to any external tools. </subproblem>

        '''
        self.role = f'''
                    You are an expert in determining whether there is a adverse effect of the drug after 
                    it interacts with th medication the patient is already taking.
                    '''
                    
        super().__init__()
        
    def summarise_and_calculate_dti_score(self, search_results, drug_name, disease_name):
        prompt = f"Does {drug_name} have any adverse interaction on {disease_name} use the following information \
            to determine adversity explain why in small description produce a score as well between\
            0 and 1 \n\n{search_results}"
            
        results = self.request(prompt)
        return results
        
        
        
    def internet_search(self, drug_name, current_diseases):
        diseases = current_diseases.split(",")
        search_detectors = DrugDiseaseInteractionDetector(self.config)
        
        responses = []
        for disease in diseases:
            search_results = search_detectors.search_drug_target(drug_name, disease)
            search_results = search_detectors.format_search_results(search_results)
            
            score = self.summarise_and_calculate_dti_score(search_results, drug_name, disease)
            responses.append(score)
            
        return "\n".join(responses)
    
    def process_subproblem(self,subproblem, user_prompt):
        process_prompt = f"The original user problem is: {self.config['base_user_prompt']}\nNow, please you solve this problem: {subproblem}"
        response = self.request(process_prompt)
        
        return response
    
    def combine_agent_results(self, problem_results, prompt):
        return "\n".join(problem_results)
        
    def process(self, drug_name, current_medication):
        current_medication = current_medication.split(",")
        prompt = "Please evaluate the interaction of drug {} with the disease list {}".format(drug_name, current_medication)
        
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