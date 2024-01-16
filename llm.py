from openai import OpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from kor.extraction import create_extraction_chain
from langchain_community.llms import Bedrock
import boto3
from data import judge_prompt_template, medical_record_schema


bedrock = boto3.client(
    service_name="bedrock",
    region_name="us-east-1",
    endpoint_url="https://bedrock.us-east-1.amazonaws.com",
)
bedrock_runtime = boto3.client(service_name="bedrock-runtime")
inference_modifier = {
    "max_tokens_to_sample": 4000,
    "temperature": 0.0,
    "top_p": 1,
}
client = OpenAI()


claude = Bedrock(
    credentials_profile_name="default", model_id="anthropic.claude-v2:1", model_kwargs=inference_modifier
)
gpt = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, top_p=1.0)

system_prompt = "You are a medical assistant and an expert in medical records. You are skilled in parsing, understanding, processing and analysing complex medical documents."

def call_model(prompt:str) -> str:
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    )
    return completion.choices[0].message.content


def call_json_parse(input_prompt: str):
    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | gpt | parser
    return chain.invoke({"query": input_prompt})

# Call GPT
def invoke_gpt(prompt:str) -> str:
    messages = [
        SystemMessage(
            content=system_prompt
        ),
        HumanMessage(
            content=prompt
        ),
    ]
    return gpt.invoke(messages).content

# Call Claude
def invoke_claude(prompt:str) -> str:
    full_prompt = f"{system_prompt} \n \n {prompt}" 
    return claude.invoke(full_prompt)


def judge(prompt: str):
    gpt_output = invoke_gpt(prompt)
    try:
        claude_output = invoke_claude(prompt)
        judge_prompt = judge_prompt_template.format(problem_statement=prompt, output_1=gpt_output, output_2=claude_output)
        output = invoke_gpt(judge_prompt)
        if 'output_1' in output.split(' ')[-1] : 
            return gpt_output
        return claude_output
    except Exception as e:
        print("Error Judging Outputs, falling back to one output")
        print(e)
        return gpt_output





def extract_data(medical_record: str):
    # Langchain LLM Initialization (Assuming you have the llm and chain initialized as per the provided documentation)

    # Create Extraction Chain
    chain = create_extraction_chain(gpt, medical_record_schema, encoder_or_encoder_class="json", input_formatter=None)

    # Run extraction
    result = chain.run((medical_record))

    # Extracted data
    return result['data']['health_record']
