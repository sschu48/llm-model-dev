# Embedding model for document upsert and query embedding

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")