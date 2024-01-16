from llm import call_json_parse, call_model
from utils import get_medical_record, print_past_treatments
from data import medical_record_pahts, recomended_treatment_prompt, past_treatment_prompt, naive_prompt, colonoscopy_guidelines


medical_record_index = 1 # TODO: Make it dynamic
medical_record = get_medical_record(medical_record_pahts[medical_record_index])


if __name__ == '__main__':
    # Get the recomended treatment
    print('Recomended Treatment:')
    treatment_output = call_model(recomended_treatment_prompt.format(medical_record=medical_record))
    print(treatment_output)
    
    # Get the prior treatments + succcess
    print('Past Treatments:')
    past_treatments = call_json_parse(past_treatment_prompt.format(medical_record=medical_record))
    print_past_treatments(past_treatments)

    if past_treatments['prior_treatments_success']:
        exit()

    #TODO: Add guideline enrichment
    # Check Qualification for treatment
    print('Treatment Qualification')
    treatment_qualification = call_model(naive_prompt.format(colonoscopy_guidelines=colonoscopy_guidelines, medical_record=medical_record))
    print(treatment_qualification)


# Naive Approach - Works, Simple


# How to improve
    
# Step 0: Run the pipeline through multiple models
# Step 1: get the doc structured
# Step 2: Determine the treatment that is recomended by the doctor: Judge
# Step 3: Determine is prior treatments improved symptoms: Judge
# Step 4: If Not: Ingest the Enriched Guidelines + Examples
# Step 5: Check if Patient Qualifies Or Suggest the additional information needed : Judge 


# Decision 1: Use RAG - May work, able to extract relevant sections but need further work to add to pipeline
# Decision 2: Use Multiple Models: Multiple LLMs allow us to avoid potential hallucinations and get higher confidence of decisions
# Decision 3: Use LLMS to structure the document: LLMs are quite good at parsing documents, since the content is similar we can use LLMs to create objects
# Decision 4: Chain of Thought (COT) reasoning when possible, research has shown models do better with COT
# Decision 5: 1 shot example for final answer.   
