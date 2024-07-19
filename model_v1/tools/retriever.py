# Setting up and using Pinecone throughout model

from langchain_pinecone import PineconeVectorStore
from models.embedding import embedding_model


def retriever():
    # init Pincone
    vectorstore = PineconeVectorStore(
        embedding=embedding_model,
        index_name="rag-retriever-v2"
    )
    retriever = vectorstore.as_retriever()
    return retriever
