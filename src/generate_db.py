from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lanchain_community.vectorestores import FAISS
from langchain_openai import OpenAIEmbeddings

DATA_PATH = "data"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.html")
    documents = loader.load()
    return documents

documents = load_documents()

print(len(documents))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(documents)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

vectore_store = FAISS(embedding_functions=embeddings_model)

documents_ids = vectore_store.add_documents(documents=all_splits)
