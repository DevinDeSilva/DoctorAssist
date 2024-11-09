from .agent import Agent

tool_list = [
    
]

class EfficacyAgent(Agent):
    def __init__(self, config, depth=1):
        self.agent_tools = tool_list
        self.config = config
        self.depth = depth
        self.reasoning_examples = ""
        self.role = f'''
                    You are an expert in determining efficacy levels of a specific drug for a disease.
                    '''
                    
        super().__init__()