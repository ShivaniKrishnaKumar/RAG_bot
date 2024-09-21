import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pickle  
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")

# os.environ["GROQ_API_KEY"] = "GROQ_API"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACE_API"
# api_key = "HUGGINGFACE_API"

EMBEDDING_FILE = 'embeddings.pkl'

# Function to process PDF and create embeddings
def process_pdf_and_create_embeddings(pdf_path):
    pdfreader = PdfReader(pdf_path)
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key,
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    document_search = FAISS.from_texts(texts, embeddings)

    # Save embeddings to a file
    with open(EMBEDDING_FILE, 'wb') as f:
        pickle.dump(document_search, f)

    return document_search

# Function to load embeddings from file if they already exist
def load_embeddings_if_exists():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, 'rb') as f:
            document_search = pickle.load(f)
        return document_search
    return None

# Load the embeddings (either load from file or create new embeddings)
document_search = load_embeddings_if_exists()

if document_search is None:
    st.write("Creating embeddings as they do not exist...")
    pdf_path = 'ILP.pdf' 
    document_search = process_pdf_and_create_embeddings(pdf_path)
    st.write("Embeddings created and saved.")
else:
    st.write("Embeddings loaded from the cache.")

llm = ChatGroq(model="llama3-8b-8192")
chain = load_qa_chain(llm, chain_type="stuff")

st.title("PDF Query Application")


query = st.text_input("Ask a question about the PDF:")

if query:
    docs = document_search.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    st.write(response)
