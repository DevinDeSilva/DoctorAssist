import traceback
import json 
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
import re

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


class Agent:
    def __init__(self):
        self.model = GPT_MODEL
        self.system_prompt = self.role
        
        if self.agent_tools and len(self.agent_tools) > 0:
            func_content = json.dumps([
                {'function_name': func['function']['name'], 'description': func['function']['description']} for func in self.agent_tools
            ], indent=4)
  
            self.system_prompt += f"\n The following tools are available for you to use: <tools>{func_content}</tools>."
            
        
        if len(self.reasoning_examples) > 0:
            self.system_prompt += f"\nFor example:{self.reasoning_examples}"
            
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
    
    def request(self, prompt):
        self.messages.append({'role': 'user', 'content': prompt})
        
        results = self.bare_request(self.messages, self.agent_tools, model=self.model)

        self.messages.append({'role': 'assistant', 'content': results})

        return results
    
    def bare_request(self, messages, tools, model):
        
        response = llm_request(messages, tools, model)

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
        
        return results
    
    def exec_func(self, response_choice):
        results = []

        if response_choice.finish_reason == "tool_calls":
            tool_calls = response_choice.message.tool_calls
            for tool_call in tool_calls:
                #print(f"[Action] Function calling...")
                pass

                try:
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    arguments = json.loads(arguments)

                    if function_name == 'multi_tool_use.parallel':
                        for sub_function in arguments['tool_uses']:
                            sub_function_name = sub_function['recipient_name'].split('.')[-1]
                            sub_arguments = sub_function['parameters']

                            result = eval(f"self.{sub_function_name}(**{sub_arguments})")

                            if result is None:
                                results.append(f"<function>{sub_function_name}</function><result>NONE</result>")
                            else:
                                results.append(f"<function>{sub_function_name}</function><result>{result}</result>")
                            
                            # Agent Level results
                            #if self.depth <= 1:
                            #    print(f"<function>{sub_function_name}</function><result>{result}</result>")
                    else:
                        result =  eval(f"self.{function_name}(**{arguments})")

                        if result is None:
                            results.append(f"<function>{function_name}</function><result>NONE</result>")
                        else:
                            results.append(f"<function>{function_name}</function><result>{result}</result>")
                        
                        # Agent Level results
                        #if self.depth <= 1:
                        #    print(f"<function>{function_name}</function><result>{result}</result>")

                except AttributeError as e:
                    print(f"Function name: {function_name}, Arguments: {arguments}")
                    print(f"Warning: {e}")
                    traceback.print_exc()
                    results.append(f"[Function]: {function_name} is called and the result is None")
                except Exception as e:
                    print(function_name)
                    print(arguments)
                    traceback.print_exc()
                    raise Exception(f"Error executing function {function_name}, Arguments: {arguments}: {e}")

                break

        return '\n'.join(results)