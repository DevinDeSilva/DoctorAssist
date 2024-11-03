import json 

class Planner:
    def __init__(self, agent_tools, config):
        self.agent_tools = agent_tools
        self.config = config
        self.role = f'''
            As a decomposition expert, you have the capability to break down a complex problem into smaller, 
            more manageable subproblems. Utilize tools to address each subproblem individually, ensuring one tool 
            per subproblem. Aim to resolve every subproblem either through a specific tool or your expertise.
            You don't need to solve it; your duty is merely to break down the problem into 
            <subproblem>subproblems</subproblem>.'''

        if self.agent_tools and len(self.agent_tools) > 0:
            func_content = json.dumps([
                {'function_name': func['function']['name'], 'description': func['function']['description']} for func in self.agent_tools
            ], indent=4)
  
            self.role += f"\n The following tools are available for you to use: <tools>{func_content}</tools>."
            
        self.system_prompt = self.role
            
        
        
    def process(prompt, examples=None):
        
        if len(examples) > 0:
            self.system_prompt += f"\nFor example:{examples}"
        pass