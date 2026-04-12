from langchain_community.document_loaders import PyMuPDFLoader
import os
from langchain.tools import tool

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

file_path = "دليل-الخدمات-إصدار-4.2-2024EN.pdf"
loader = PyMuPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Maximum of 500 characters per chunk
    chunk_overlap=100,  # Overlapping 100 characters to preserve context
    length_function=len,  # Determines chunk size based on character length
    is_separator_regex=False,  # The separator used for splitting is not a regex pattern
)


# Split documents into smaller chunks
split_documents = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Set directory for persistent storage
persist_directory = "uae_visa_index"

# Store documents in ChromaDB
vectorstore = Chroma.from_documents(
    documents=split_documents,  # Ensure splited_documents contains valid data
    embedding=embeddings,
    persist_directory=persist_directory
)

# Persist the database to disk
vectorstore.persist()  
print("✅ Data successfully stored in ChromaDB!")

# Reload the vector store for retrieval
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
print("🔄 ChromaDB reloaded successfully!")
