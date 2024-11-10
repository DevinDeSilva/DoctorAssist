from .agent import Agent

tool_list = [
    
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
        

    
        