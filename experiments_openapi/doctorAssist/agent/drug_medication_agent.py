from .agent import Agent

tool_list = [
    
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