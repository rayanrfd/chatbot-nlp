import faiss
import bs4
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

loader = DirectoryLoader("data", glob="*.txt")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  
    chunk_overlap=50,  
    length_function=len,
    separators = ["\n\n","\n",".","?","!"," ",""]
)

if __name__=="__main__":
    docs = loader.load()
    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.add_documents(documents=all_splits)
    vector_store.save_local("faiss_index")
