import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import PyMuPDFLoader
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

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

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)

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
