from .agent import Agent
import re
 
def subproblem_extraction(content):
    subproblems = re.findall(r"<subproblem>(.*?)</subproblem>", content)
    subproblems = [subproblem.strip() for subproblem in subproblems]
    
    return subproblems

class DecompositionAgent(Agent):
    def __init__(self, config, tools, depth=1):
        self.agent_tools = tools
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