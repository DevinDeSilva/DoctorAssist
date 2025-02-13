from .agent import Agent
import re
 
def subproblem_extraction(content):
    subproblems = re.findall(r"<subproblem>(.*?)</subproblem>", content)
    subproblems = [subproblem.strip() for subproblem in subproblems]
    
    return subproblems

class DecompositionAgent(Agent):
    def __init__(self, config, tools, reasoning_examples, depth=1):
        self.agent_tools = tools
        self.config = config
        self.depth = depth
        self.role = f'''
            As a decomposition expert, you have the capability to break down a complex problem into smaller, 
            more manageable subproblems. Utilize tools to address each subproblem individually, ensuring one tool 
            per subproblem. Aim to resolve every subproblem either through a specific tool or your expertise.
            You don't need to solve it; your duty is merely to break down the problem into 
            <subproblem>subproblems</subproblem>.'''
            
        self.reasoning_examples = reasoning_examples
        
        super().__init__()
        
        
    def process(self,prompt):
        
        response = self.request(prompt)
        subproblems = subproblem_extraction(response)
            
        return subproblems