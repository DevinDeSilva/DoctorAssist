import json 
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
import re
import random

from .agent import Agent

load_dotenv()
        
def subproblem_extraction(content):
    subproblems = re.findall(r"<subproblem>(.*?)</subproblem>", content)
    subproblems = [subproblem.strip() for subproblem in subproblems]
    
    return subproblems

client = OpenAI()
GPT_MODEL = 'gpt-4o-mini'


tool_list = [
    {
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
        
        {
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
        
        
        {
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
        
        {
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
        
        
]

class DecompositionAgent(Agent):
    def __init__(self, config, depth=1):
        self.agent_tools = tool_list
        self.config = config
        self.depth = depth
        self.role = f'''
            As a decomposition expert, you have the capability to break down a complex problem into smaller, 
            more manageable subproblems. Utilize tools to address each subproblem individually, ensuring one tool 
            per subproblem. Aim to resolve every subproblem either through a specific tool or your expertise.
            You don't need to solve it; your duty is merely to break down the problem into 
            <subproblem>subproblems</subproblem>.'''
            
        self.reasoning_examples = '''
            Question: How can we predict whether this drug can treat the disease?
            Answer: <subproblem> To understand the drug's not toxic to the patient we need to consult the toxicity agent for information in toxicity risk. </subproblem>
            <subproblem> To evaluate the drug's efficacy against diseases, it is essential to request information on the drug's effectiveness from the Efficacy Agent. Obtain details about the drug, including its description, pharmacological indications, absorption, volume of distribution, metabolism, route of elimination, and toxicity. </subproblem>
            <subproblem> Ask the drug_disease agent to assess the whether the drug has any adverse effect on the diseases that the patient currently has stated in the Patient Profile</subproblem>
            <subproblem> Ask the drug_drug_age agent to assess the whether the drug has any adverse interaction with the drugs that the patient currently is taking as stated in the Patient Profile</subproblem>

            Question: How can I evaluate the toxicity of the drug aspirin on the disease diabetes?
            <subproblem> To assess the risk associated with the drug, we must use the "get_drug_toxicity" tool. </subproblem>
            <subproblem> Provide insights into the toxicity of both the drug and the disease based on your expertise, without resorting to any external tools. </subproblem>

            Question: How can I evaluate the efficacy of the drug aspirin on the disease diabetes?
            Answer:
            <subproblem> To understand the drug's structure, obtaining the SMILES notation of the drug is necessary. </subproblem>
            <subproblem> Assessing the drug's effectiveness requires retrieving information from the DrugBank database. </subproblem>
            <subproblem> To evaluate the drug's impact on the disease, we must obtain the pathway linking the drug to the disease from the Hetionet Knowledge Graph by using the retrieval_hetionet tool. </subproblem>
            <subproblem> Offer insights on the drug and disease based on your expertise, without resorting to any external tools. </subproblem>
            
            Question: How can I evaluate any adverse effect of the drug aspirin on the Current Diseases in the Patient Profile?
            <subproblem> To assess the adverse effect, we first must select the Current Diseases in the Patient Profile. </subproblem>
            <subproblem> To evaluate the adverse effect of the drug, we must use the "get_drug_disease_evaluation" tool. </subproblem>
            <subproblem> Provide insights into the adverse effect of the drug on the Current Diseases in the Patient Profile, without resorting to any external tools. </subproblem>

            Question: How can I evaluate any adverse interaction of the drug aspirin on the Current Medication in the Patient Profile?
            <subproblem> To assess the adverse interaction, we first must select the Current Medication in the Patient Profile. </subproblem>
            <subproblem> To evaluate the adverse interaction of the drug, we must use the "get_drug_medication_evaluation" tool. </subproblem>
            <subproblem> Provide insights into the adverse interaction of the drug on the Current Medication in the Patient Profile, without resorting to any external tools. </subproblem>

        '''
        
        super().__init__()

        
        
        
        
    def process(self,prompt):
        
        response = self.request(prompt)
        subproblems = subproblem_extraction(response)
            
        return subproblems



class Planner(Agent):
    def __init__(self, config, depth=1):
        self.agent_tools = tool_list
        self.config = config
        self.depth = depth
        self.reasoning_examples = ""
        self.role = f'''
                    You are an expert in assisting doctors to choose drugs for patients.
                    '''
                    
        super().__init__()
        
    def process_subproblem(self,subproblem, user_prompt):
        process_prompt = f"The original user problem is: {user_prompt}\nNow, please you solve this problem: {subproblem}"
        response = self.request(process_prompt)
        
        return response
        
    def combine_agent_results(self, problem_results, prompt):
        #change system prompt
        
        #create the combination results.
        
        #run request using request_custom_messages
        messages = []
        
        system_prompt = f''' 
            You are an expert in Assisting Doctors choose drug for a patients disease. Based on your own knowledge and the sub-problems have solved, please solve the user's problem and provide the reason.
            First, Analysis the user's problem.
            Second, present the final result of the user's problem in <final_result></final_result>, for a binary problem, it is a value between 0 to 1. You must include the exact probability within the <final_result></final_result> tags, e.g., 'The probability of the drug is effective against the disease with no adverse effect to patient<final_result>0.8</final_result>.'
            Third, explain the reason step by step.
            Noted, you must include the exact probability within the <final_result></final_result> tags.
        '''
    
        # The following examples are essential for your understanding, offering significant learning opportunities. By analyzing these examples, including sub-problems and labels, you will learn how to make accurate predictions.
        # Each example is within the <example></example> tags.
        # Examples:
        # {fewshot_examples}

        messages.append({ "role": "system", "content": system_prompt})
        messages.append({ "role": "user", "content": f"The original user problem is: {prompt}"})
        messages.append({ "role": "user", "content": f"The subproblems have solved are: {problem_results}"})
        messages.append({ "role": "user", "content": "Please solve the user's problem and provide the reason."})
        
        results = self.bare_request(messages, [], self.model)
        
        return results

            
    
    def toxicity_agent(self, drug_name):
        
        return f"{random.random()}"
    
    def efficacy_agent(self, drug_name, disease_name):
        return f"{random.random()}"
    
    def drug_disease_agent(self, drug_name, current_diseases):
        return f"{random.random()}"
    
    def drug_medication_agent(self, drug_name, current_medications):
        return f"{random.random()}"
        
        
        
    def process(self,subproblems, prompt):
        responses = []
        
        for subproblem in subproblems:
            response = self.process_subproblem(subproblem, prompt)
            responses.append(response)
            
        final_results = self.combine_agent_results(responses, prompt)
            
        return final_results
        