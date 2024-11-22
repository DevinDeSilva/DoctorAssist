import json 
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
import re
import random

from .agent import Agent
from .decomposer import DecompositionAgent
from .toxicity_agent import ToxicityAgent
from .drug_medication_agent import DrugMedicationAgent
from .drug_disease_agent import DrugDiseaseAgent
from .efficacy_agent import EfficacyAgent

load_dotenv()

client = OpenAI()
GPT_MODEL = 'gpt-4o-mini'


tool_list = [
    {
            "type": "function",
            "function": {
                "name": "toxicity_agent",
                "description": "To understand the toxicity of the drug for a human, consult the Toxicity Agent. Given drug name, return the toxicity information of the drug",
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




class Planner(Agent):
    def __init__(self, config, depth=1):
        self.agent_tools = tool_list
        self.config = config
        self.depth = depth
        self.reasoning_examples = '''
        
            Question: The patient is experiencing a disease will the use of a drug be effictive against him and not harmful 
            to the patient.
             
            Answer: 
            To solve this problem, we need to break it down into smaller subproblems in the aspects of toxicity, efficacy, drug interactions with current diseases and drug interactions with
            current medication.
            <subproblem> To understand the toxicity of the drug to the patient and a human we need to consult the toxicity agent for information in toxicity. </subproblem>
            <subproblem> To evaluate the drug's efficacy against diseases, it is essential to request information on the drug's effectiveness from the Efficacy Agent. Obtain details about the drug, including its description, pharmacological indications, absorption, volume of distribution, metabolism, route of elimination. </subproblem>
            <subproblem> Ask the drug_disease agent to assess the whether the drug has any adverse effect on the diseases that the patient currently has stated in the Patient Profile</subproblem>
            <subproblem> Ask the drug_medication agent to assess the whether the drug has any adverse interaction with the drugs that the patient currently is taking as stated in the Patient Profile</subproblem>

        '''
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
        print(drug_name)
        tox_agent = ToxicityAgent(self.config)
        results = tox_agent.process(drug_name)
        
        return results
    
    def efficacy_agent(self, drug_name, disease_name):
        print(drug_name, disease_name)
        efficacy_agent = EfficacyAgent(self.config)
        results = efficacy_agent.process(drug_name, disease_name)
        return results
    
    def drug_disease_agent(self, drug_name, current_diseases):
        print(drug_name, current_diseases)
        drug_disease_agent = DrugDiseaseAgent(self.config)
        results = drug_disease_agent.process(drug_name, current_diseases)
        return results
    
    def drug_medication_agent(self, drug_name, current_medications):
        print(drug_name, current_medications)
        drug_med_agent = DrugMedicationAgent(self.config)
        results = drug_med_agent.process(drug_name, current_medications)
        return results
        
        
        
    def process(self, prompt):
        self.config["base_user_prompt"] = prompt
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
        