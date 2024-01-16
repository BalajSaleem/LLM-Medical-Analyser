from langchain.text_splitter import RecursiveCharacterTextSplitter
import random
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def chunk_data():
    # Use the recursive character splitter
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        is_separator_regex=True,
    )

    # Perform the splits using the splitter
    data_splits = recur_splitter.split_documents(pages)
    print(random.choice(data_splits).page_content)
    print(len(data_splits))


def create_vector_db():
    ### Using embeddings by MPNET
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    model_name = "multi-qa-MiniLM-L6-cos-v1"
    encode_kwargs = {"normalize_embeddings": False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name, encode_kwargs=encode_kwargs
    )
    db = Chroma.from_documents(data_splits, hf_embeddings)