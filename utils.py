from langchain_community.document_loaders import PyPDFLoader

def load_document_pages(path: str):
    loader = PyPDFLoader(path)
    return loader.load_and_split()

def get_medical_record(path: str):
    pages = load_document_pages(path)
    return '\n'.join([page.page_content for page in pages]) 

def print_past_treatments(past_treatments):
    if past_treatments['prior_treatments_success']:
        print('Prior Treatment has shown success')
        print('Evidence:')
        print(past_treatments['evidence'])
    else:
        print('Prior Treatments have been ineffective')
        print('Evidence:')
        print(past_treatments['evidence'])




