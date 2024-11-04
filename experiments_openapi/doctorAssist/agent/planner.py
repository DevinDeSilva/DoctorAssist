import json 
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI

load_dotenv()
client = OpenAI()
GPT_MODEL = 'gpt-4o-mini'

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(1))
def llm_request(messages, tools=None, model=GPT_MODEL):
    try:
        if tools and len(tools) > 0:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )

        return response
    except Exception as e:
        try:
            print(e)
            """
            if tools and len(tools) > 0:
                response = client.chat.completions.create(
                    model='gpt-4-turbo',
                    messages=messages,
                    tools=tools
                )
            else:
                response = client.chat.completions.create(
                    model='gpt-4-turbo',
                    messages=messages
                )

            return response
            """
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(messages)
            print(tools)
            print(f"Exception: {e}")
            raise e


class Planner:
    def __init__(self, agent_tools, config, model=GPT_MODEL, depth=1):
        self.agent_tools = agent_tools
        self.config = config
        self.model = model
        self.depth = depth
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
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        
        
    def process(self,prompt, examples=""):
        self.messages.append({'role': 'user', 'content': prompt})
        
        if len(examples) > 0:
            self.system_prompt += f"\nFor example:{examples}"
        
        self.messages.append({'role': 'user', 'content': prompt})
        
        response = llm_request(self.messages, [], self.model)

        return response
        """
        results = []
        for choice in response.choices:
            if choice.finish_reason in ['tool_calls', 'function_call']:
                results.append(self.exec_func(choice))
            elif choice.finish_reason == 'stop':
                results.append((choice.message.content))
            elif choice.finish_reason == 'content_filter':
                raise Exception("Content filter triggered.")
            elif choice.finish_reason == 'length':
                raise Exception("Max token length reached.")
            else:
                raise Exception(f"Unknown finish reason: {choice.finish_reason}")
        
        results = '\n'.join(results)

        self.messages.append({'role': 'assistant', 'content': results})
        """

        return results