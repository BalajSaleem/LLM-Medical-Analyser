from kor.nodes import Object, Text, Number
from langchain.prompts import PromptTemplate


colonoscopy_guidelines = """
[45378] Colonoscopy, flexible; diagnostic 
 
•  Colorectal cancer screening, as indicated by 1 or more of the following: 
o  Patient has average-risk or higher, as indicated by ALL of the following 
§  Age 45 years or older 
§  No colonoscopy in past 10 years 
o  High risk family history, as indicated by 1 or more of the following: 
§  Colorectal cancer diagnosed in one or more first-degree relatives of any age and ALL of the following: 
•  Age 40 years or older 
•  Symptomatic (eg, abdominal pain, iron deficiency anemia, rectal bleeding) 
§  Family member with colonic adenomatous polyposis of unknown etiology 
o  Juvenile polyposis syndrome diagnosis indicated by 1 or more of the following: 
§  Age 12 years or older and symptomatic (eg, abdominal pain, iron deficiency anemia, rectal bleeding, 
telangiectasia) 
§  Age younger than 12 years and symptomatic (eg, abdominal pain, iron deficiency anemia, rectal bleeding, 
telangiectasia) 
"""

medical_record_pahts = [
    'task\medical-record-1.pdf',
    'task\medical-record-2.pdf',
    'task\medical-record-3.pdf'
]

# Schema Definition
medical_record_schema = Object(
    id="health_record",
    description="Patient Health Record",
    attributes=[
        Object(
            id="patient_information",
            description="Patient Information",
            attributes=[
                Text(id="patient_name", description="Patient Name"),
                Text(id="dob", description="Date of Birth"),
                Text(id="mrn", description="Medical Record Number"),
                Text(id="sex", description="Gender"),
                Text(id="address", description="Address"),
                Text(id="contact_number", description="Contact Number"),
            ],
        ),
        Object(
            id="presenting_complaint",
            description="Presenting Complaint",
            attributes=[
                Text(id="symptoms", description="Symptoms"),
                Text(id="duration", description="Duration"),
            ],
        ),
        Object(
            id="medical_history",
            description="Medical History",
            attributes=[
                Text(id="family_history", description="Family History"),
                Object(
                    id="personal_medical_history",
                    description="Personal Medical History",
                    attributes=[
                        Object(id="Diagnosis", description="Past Patient Diagnosis",
                             attributes=[
                                Text(id="Name", description="Name of the Diagnosis"),
                                Text(id="Date", description="Date of the Diagnosis"),
                                Text(id="Details", description="Details of the Diagnosis"),
                             ]),
                        Object(id="Treatments", description="Past Treatments",
                            attributes=[
                                Text(id="Name", description="Name of the Treatment"),
                                Text(id="Date", description="Date of the Treatment"),
                                Text(id="Details", description="Details of the Treatment"),
                             ]),
                    ],
                ),
                Text(id="medications", description="Medications"),
                Text(id="allergies", description="Allergies"),
            ],
        ),
        Object(
            id="vitals",
            description="Vitals",
            attributes=[
                Number(id="height", description="Height"),
                Number(id="weight", description="Weight"),
                Number(id="bmi", description="BMI"),
                Number(id="pulse", description="Pulse"),
            ],
        ),
        Text(id="Diagnostic Tests", description="Diagnostic Tests"),
        Text(id="history", description="History"),
        Text(id="clinical_impression", description="Clinical Impression"),
        Text(id="laboratory_results", description="Laboratory Results"),
        Object(
            id="plan",
            description="Plan",
            attributes=[
                Text(id="planned treatments", description="Treatments planned"),
                Text(id="pre_procedure_instructions", description="Pre-Procedure Instructions"),
                Text(id="post_procedure_care", description="Post-Procedure Care"),
                Text(id="consent", description="Consent"),
            ],
        ),
        Text(id="follow_up", description="Follow-Up"),
    ],
)

example_analysis_additional_information_json = """
{
    "analysis": [
      {
        "type": "Colorectal Cancer Screening",
        "details": [
          "Patient is 57 years old (satisfying age criteria).",
          "One colonoscopy found in the past 10 years."
        ],
        "additionalInformation": "Not Required",
        "qualifies": False,
        "reason": "The patient does not Qualify through the Colorectal Cancer Screening Criteria, since they have had 2 colonoscopies in the past 10 years."
      },
      {
        "type": "High-Risk Family History",
        "details": [
          "Father had colorectal cancer (meeting family history criteria).",
          "Age at diagnosis is not specified.",
          "Symptoms of the father are not fully described.",
          "No information is specified regarding Family member with colonic adenomatous polyposis of unknown etiology."
        ],
        "additionalInformation": [
          "Request details on the age and symptoms of Patients's father during colorectal cancer diagnosis.",
          "Request details regarding adenomatous polyposis of unknown etiology in first-degree relatives."
        ],
        "qualifies": False,
        "reason": "With the current information, the patient does not Qualify through the High-Risk Family History Criteria. While the father had colorectal cancer, age of father's diagnosis is not specified."
      },
      {
        "type": "Juvenile Polyposis Syndrome Diagnosis",
        "details": [
          "No indication of juvenile polyposis syndrome in the medical record."
        ],
        "additionalInformation": [
          "Inquire about any history or symptoms related to juvenile polyposis syndrome."
        ],
        "qualifies": False
        "reason": "With the current information, the patient does not Qualify through the Juvenile Polyposis Syndrome Diagnosis Criteria. No indication of Juvenile Polyposis Syndrome was found."
      }
    ],
    "qualifies": False
    "reason": "The Patient Does Not Qualify For Treatment as none of the criteria have been fulfilled"
  }
"""

example_analysis_qualifies_json = """
{
  "analysis": {
    "ColorectalCancerScreening": {
      "details": [
        "Patient is 40 years old.",
        "No clear documentation of a colonoscopy in the past 10 years."
      ],
      "additionalInformation": [
        "Request clarification on any undisclosed colonoscopies."
      ],
      "qualifies": False,
      "reason": "The patient does not Qualify through the Colorectal Cancer Screening Criteria. The patient's age is lower than 45."
    },
    "HighRiskFamilyHistory": {
      "details": [
        "Father had colorectal cancer (meeting family history criteria).",
        "Father's age at diagnosis 55.",
        "Symptoms of the father include: abdominal pain and rectal bleeding ."
      ],
      "additionalInformation": [
        "No Additional Information Required"
      ],
      "qualifies": True,
      "reason": "The Patient Qualifies through the High-Risk Family History Criteria. The father had colorectal cancer, was symptomatic, and the age at diagnosis was greater than 40."
    },
    "JuvenilePolyposisSyndromeDiagnosis": {
      "details": [
        "No indication of juvenile polyposis syndrome in the medical record."
      ],
      "additionalInformation": [
        "Inquire about any history or symptoms related to juvenile polyposis syndrome."
      ],
      "qualifies": False,
      "reason": "With the current information, the patient does not Qualify through the Juvenile Polyposis Syndrome Diagnosis Criteria. No indication of Juvenile Polyposis Syndrome was found."
    },
    "qualifies": True
    "reason": "The Patient Qualifies For Treatment as one of the criteria i.e. \"High-Risk Family History\" is fulfilled."
  }
}

"""


example_analysis_additional_information = """
**Example of Analysis**

- **Colorectal Cancer Screening:**
  - Patient is 57 years old (satisfying age criteria).
  - One a colonoscopy found in the past 10 years.
  - *Additional Information:*
    - Not Required

    Qualification: The patient does not Qualify through the Colorectal Cancer Screening Criteria, since they have had 2 conoscopies in the past 10 years.


- **High-Risk Family History:**
  - Father had colorectal cancer (meeting family history criteria).
  - Age at diagnosis is not specified.
  - Symptoms of the father are not fully described.
  - No information is specified regarding Family member with colonic adenomatous polyposis of unknown etiology.
  - *Additional Information:*
    - Request details on the age and symptoms of Patients's father during colorectal cancer diagnosis.
    - Request details regarding adenomatous polyposis of unknown etiology in first degree relatives 

  Qualification: With the current information, the patient does not Qualify through the High-Risk Family History Criteria. While the father had colorectal cancer, age of father's diagnosis is not specified.

- **Juvenile Polyposis Syndrome Diagnosis:**
  - No indication of juvenile polyposis syndrome in the medical record.
  - *Additional Information:*
    - Inquire about any history or symptoms related to juvenile polyposis syndrome.

  Qualification: With the current information, the patient does not Qualify through the Juvenile Polyposis Syndrome Diagnosis Criteria. No indication of Juvenile Polyposis Syndrome was found. 

"""

example_analysis_qualifies = """
**Example of Analysis**

- **Colorectal Cancer Screening:**
  - Patient is 40 years old.
  - No clear documentation of a colonoscopy in the past 10 years.
  - *Additional Information:*
    - Request clarification on any undisclosed colonoscopies.
  
  Qualification: The patient does not Qualify through the Colorectal Cancer Screening Criteria. The patient's age is lower than 45.


- **High-Risk Family History:**
  - Father had colorectal cancer (meeting family history criteria).
  - Father's age at diagnosis 55.
  - Symptoms of the father include: abdominal pain and rectal bleeding .
  - *Additional Information:*
    - No Additional Information Required

    Qualification: The Patient Qualifies through the High-Risk Family History Criteria. The father had colorectal cancer, was symptomatic and the age at diagnosis was greater than 40.


- **Juvenile Polyposis Syndrome Diagnosis:**
  - No indication of juvenile polyposis syndrome in the medical record.
  - *Additional Information:*
    - Inquire about any history or symptoms related to juvenile polyposis syndrome.

    Qualification: With the current information, the patient does not Qualify through the Juvenile Polyposis Syndrome Diagnosis Criteria. No indication of Juvenile Polyposis Syndrome was found. 
"""

enriched_guidelines = """
Structured Guidelines for Colonoscopy, Flexible; Diagnostic:

1. **Colorectal Cancer Screening:**
   - Indicated for individuals with average-risk or higher, meeting ALL of the following criteria:
     - Age 45 years or older.
     - No colonoscopy in the past 10 years.

2. **High-Risk Family History:**
   - Indicated for individuals with a high-risk family history, meeting 1 or more of the following criteria:
     - Colorectal cancer diagnosed in one or more first-degree relatives of any age, meeting ALL of the following:
       - Age 40 years or older.
       - Symptomatic (e.g., abdominal pain, iron deficiency anemia, rectal bleeding).
     - Family member with colonic adenomatous polyposis of unknown etiology.

3. **Juvenile Polyposis Syndrome Diagnosis:**
   - Indicated for individuals with a juvenile polyposis syndrome diagnosis, meeting 1 or more of the following criteria:
     - Age 12 years or older and symptomatic (e.g., abdominal pain, iron deficiency anemia, rectal bleeding, telangiectasia).
     - Age younger than 12 years and symptomatic (e.g., abdominal pain, iron deficiency anemia, rectal bleeding, telangiectasia).

If the patient fulfills ANY of the Major Factors above (i.e. Colorectal Cancer Screening, High-Risk Family History OR Juvenile Polyposis Syndrome Diagnosis), they would qualify for the conoscopy. 
"""


enriched_guidelines_2 = """
Set 1: Colorectal Cancer Screening
Criteria:
Age 45 years or older.
No colonoscopy in the past 10 years.
Set 2: High-Risk Family History
Criteria:
Colorectal cancer diagnosed in one or more first-degree relatives of any age, meeting ALL of the following:
Age 40 years or older.
Symptomatic (e.g., abdominal pain, iron deficiency anemia, rectal bleeding).
Family member with colonic adenomatous polyposis of unknown etiology.
Set 3: Juvenile Polyposis Syndrome Diagnosis
Criteria:
Age 12 years or older and symptomatic (e.g., abdominal pain, iron deficiency anemia, rectal bleeding, telangiectasia).
Age younger than 12 years and symptomatic (e.g., abdominal pain, iron deficiency anemia, rectal bleeding, telangiectasia).
Qualification for Colonoscopy
If the patient fulfills ANY of the Major Factors above (i.e., Colorectal Cancer Screening, High-Risk Family History OR Juvenile Polyposis Syndrome Diagnosis), they would qualify for the colonoscopy.
"""

judge_prompt_template = PromptTemplate.from_template(
    """
    We are going to analyse the accuracy of two outputs given a problem statement.

    You are given the following problem statement.

    {problem_statement}

    You are also given the output from two experts: 
    
    Expert Output 1: {output_1}

    Expert Output 2: {output_2}

    Analyse each output deeply and determine which output is more accurate for the given the problem statement.
    Think step by step.

    You must write "output_1" if expert output 1 is more accurate. Similarly if output 2 is more accurate you must write "output_2" 
    """
)

recomended_treatment_prompt = PromptTemplate.from_template("""You are given a patient's health record below. It contains various sections on medical and clinical information on to the patient

    {medical_record}

    Use the health record to determine the treatment(s) recomended by the doctor and extract any CPT code(s) if present. 
    """)

past_treatment_prompt = PromptTemplate.from_template("""
    You are given a patient's health record below. It contains various sections on medical and clinical information on to the patient

    {medical_record}

    Analyse the record. Take the following steps in order:
    1. Identify the patient's current symptoms
    2. Use the health record to identify any prior treatments if present.
    3. Determine if the prior treatments improved the patient's current symptoms. 


    Use your analysis to create the output below:

    prior_treatments_success: Determine if the prior treatments improved the patient's current symptoms. (boolean)
    evidence: Present evidence from the medical record for your answer to "prior_treatment_failed"
    """)

enrichment_prompt = PromptTemplate.from_template(""" You are given the following medical guidelines:

    {colonoscopy_guidelines}

    These guidelines describe criteria for a treatment. The guidelines will be used along with patients' health records to determine if a treatment is needed. 

    Use these guidelines to create sets of potential evidences to look for in the patients records that would qualify the patients for the treatment.   
    """)

naive_prompt = PromptTemplate.from_template(""" You are given the following medical guidelines. These guidelines describe criteria for a treatment:

    {colonoscopy_guidelines}

    You are also given a patient's health record below.

    {medical_record}

    You must use the guidelines and patient's health record to determine if the patient qualifies for the treatment described in the guidelines.

    For each point in the guidelines, determine if the patient qualifies or not based on their health record. 

    In your output you must go through each point in the guidelines and do one of the following:
    * Refer to qualifying evidence in the document if present.
    * Mention evidence is missing to qualify.

    At the end of your analysis, you should make a recomendation.
    If qualifying evidence is found, the patient must be recomended for the treatment.
    If qualifying evidence is not found, you must list all the additional peices of information that may be required to make the decision.

    You must not assume or make up information about the patient.
    """)


one_shot_prompt = PromptTemplate.from_template(
    """ You are given the following medical guidelines. These guidelines describe criteria for a treatment:

    {colonoscopy_guidelines}

    You are also given a patient's health record below.

    {medical_record}

    You must use the guidelines and patient's health record to determine if the patient qualifies for the treatment described in the guidelines.

    For each point in the guidelines, determine if the patient qualifies or not based on their health record. 

    In your output you must go through each point in the guidelines and do one of the following:
    * Refer to qualifying evidence in the document if present.
    * Mention evidence is missing to qualify.

    At the end of your analysis, you should make a recomendation.
    If qualifying evidence is found, the patient must be recomended for the treatment.
    If qualifying evidence is not found, you must list all the additional peices of information that may be required to make the decision.

    You must not assume or make up information about the patient.

    The following are example outputs:

    Example 1 - Patient does not qualify, additional information requested: {example_analysis_additional_information}
    Example 2 - Patient qualifies: {example_analysis_qualifies}

    """
)