from .agent import Agent

tool_list = [
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
        }
]

class DrugMedicationAgent(Agent):
    def __init__(self, config, depth=1):
        self.agent_tools = tool_list
        self.config = config
        self.depth = depth
        self.reasoning_examples = ""
        self.role = f'''
                    You are an expert in determining whether there is a adverse effect of the drug after 
                    it interacts with th medication the patient is already taking.
                    '''
                    
        super().__init__()
        
    def database_search(self, drug_name, current_medications):
        pass