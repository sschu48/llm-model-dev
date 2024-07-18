# Agentic RAG using llama-agents
# July 2024
# Version 2.0 --> iteration from "agentic_rag.py"


# First, connect to Pinecone to retriever context
# This will be our retriever

import os
from llama_index.core.schema import NodeWithScore
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List

class PineconeRetriever(BaseRetriever):
    """
    Retriever over a pinecone vector store
    """

    def __init__(
            self, 
            vector_store: PineconeVectorStore,
            embed_model: Any,
            query_mode: str = "default",
            similarity_top_k: int = 2,
    ) -> None:
        """Init params"""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve"""
        query_embedding = embed_model.get_query_embedding(query_str)
        vector