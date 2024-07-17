# Agentic RAG using llama-agents
# July 2024
# Version 1.0


# First, connect to Pinecone to retriever context
# This will be our retriever

import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore

def pinecone_init(index_name, pinecone_api_key):
    # initialize connection to pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # initialize index
    try:
        pinecone_index = pc.Index(index_name)
        print("Index loaded successfully")
    except Exception as e:
        print(f"error occurred initialize Pinecone index; {e}")

    return pinecone_index

def vectorstore_init(pinecone_index):
    # initialize vectorstore
    try:
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    except Exception as e: 
        print(f"error occurrred setting up vector store: {e}")

    return vector_store