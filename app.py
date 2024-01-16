from llm import call_json_parse, call_model, extract_data, invoke_gpt, judge
from utils import get_medical_record, print_past_treatments
from data import medical_record_pahts, one_shot_prompt, past_treatment_prompt, enriched_guidelines_2, naive_prompt, example_analysis_qualifies, example_analysis_additional_information
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process medical records and recommend treatment.')
    parser.add_argument('--medical_record_path', type=str, required=True, help='Path to the medical record file')
    parser.add_argument('--judge_llms', type=bool, required=False, default=False, help='Should Multiple LLMs be chained and best output returned')
    return parser.parse_args()


if __name__ == '__main__':
    # Get the medical Record
    args = parse_arguments()
    medical_record_path = args.medical_record_path
    judge_llms = args.judge_llms
    # medical_record_path = 'task\medical-record-2.pdf'
    medical_record = get_medical_record(medical_record_path)
    print('Medical Record Loaded')

    # Step 1: get the doc structured
    structured_medical_record = extract_data(medical_record)

    print('Recomended Treatment:')
    treatment_output = structured_medical_record['plan']['planned treatments']
    print(treatment_output)
    
    # Get the prior treatments + succcess
    print('Past Treatments:')
    past_treatments = call_json_parse(past_treatment_prompt.format(medical_record=medical_record))
    print_past_treatments(past_treatments)

    # May need a error handling wrapper
    if past_treatments['prior_treatments_success']:
        exit()

    # Check Qualification for treatment
    print('Treatment Qualification:')
    prompt = one_shot_prompt.format(example_analysis_qualifies=example_analysis_qualifies, example_analysis_additional_information=example_analysis_additional_information,
                                              colonoscopy_guidelines=enriched_guidelines_2, medical_record=structured_medical_record)
    # prompt = naive_prompt.format(colonoscopy_guidelines=enriched_guidelines_2, medical_record=structured_medical_record)
    treatment_qualification = ''
    if judge_llms:
        print('Pooling Multiple LLM Outputs...')
        treatment_qualification = judge(prompt)
    else:
        treatment_qualification = invoke_gpt(prompt)
    print(treatment_qualification)




