Certainly! Here's a basic README template for your Python app:

# Medical Doc Analyser

This Python app is designed to Ingest and Analyse medical records and approve them for treatment.

## How It Works

* `experiments` is notebook for different approaches tried
* `app.py` contains the main logic
* `simple_app.py` contains the naive logic to get the prototype working
* `utils.py` continas helper functions
* `llm.py` contains code for interacting with models
* `task` directory contains the medical data

    
* Step 1: get the doc structured
* Step 2: Determine the treatment that is recomended by the doctor
* Step 3: Determine is prior treatments improved symptoms
* Step 4: If Not: Ingest the Enriched Guidelines + Examples
* Step 5: Check if Patient Qualifies Or Suggest the additional information needed : Judge 


## Design Decisions:

* Decision 1: Use RAG - May work, able to extract relevant sections but need further work to add to pipeline
* Decision 2: Use Multiple Models: Multiple LLMs (currently claude + openai) allow us to avoid potential hallucinations and get higher confidence of decisions
* Decision 3: Use LLMS to structure the document: LLMs are quite good at parsing documents, since the content is similar we can use LLMs to create objects
* Decision 4: Chain of Thought (COT) reasoning when possible, research has shown models do better with COT
Decision 5: 1 shot example for final answer.   

## Future Plans
* The following are some short comings / remaining improvements that could be addressed given more time

* Use a model fine tuned on clinical data
* Improve RAG to fetch relevant Data (See experiments.ipynb) for current state
* Use APE, Automated Prompt Engineering based on a reward for the model when the right output is returned
* Scale -> Dockerize + Plug to AWS Lambda - Use Opensearch / Elastic Search for RAG


## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- [OpenAI API Key](https://openai.com/blog/openai-api)
- (Optional) For Judging LLMs, using Claude, [AWS Setup](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html). Please ensure claude in bedrock is enabled in the account, the region is us-east-1.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/balajsaleem/MedicalAnalyser.git
   ```

2. Navigate to the project directory:

   ```bash
   cd MedicalAnalyser
   ```

3. Create a Conda environment:

   ```bash
   conda create --name your-environment-name python=3.10
   ```

4. Activate the Conda environment:

   ```bash
   conda activate your-environment-name
   ```

5. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the main application:

```bash
python app.py --medical_record_path=task/medical-record-3.pdf
```

## Arguments
* **medical_record_path**: The path to the medical record to ingest - string
* **judge_llms**: Flag to enable LLM output pooling + judgement

