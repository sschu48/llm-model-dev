# retrieve relevant documents from Pinecone
# embed user query and perform similarity search between query and vector db
# return relevant documents

from pinecone import Pinecone
from typing import List, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = "rag-retriever-v2"
        self._initialize_pinecone()

    def _initialize_pinecone(self):
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
            
            if self.index_name not in self.pc.list_indexes().names():
                raise ValueError(f"Index '{self.index_name}' does not exist in Pinecone")
            
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            self.index = None

    def retrieve(self, query: str, conversation_history: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.index:
            raise RetrievalError("Pinecone index not initialized. Cannot retrieve documents.")

        try:
            # Combine the current query with the conversation history for a more contextual search
            contextual_query = f"{conversation_history}\n\nCurrent question: {query}"
            query_embedding = self.embedder.encode([contextual_query])[0]
            
            results = self.index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
            
            retrieved_docs = []
            for match in results['matches']:
                doc = {
                    'id': match['id'],
                    'score': float(match['score']),
                    'title': match['metadata'].get('title', ''),
                    'page': match['metadata'].get('page', ''),
                    'source': match['metadata'].get('source', '')
                }
                retrieved_docs.append(doc)
            
            logger.debug(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error retrieving documents from Pinecone: {e}")
            raise RetrievalError(f"Error retrieving documents from Pinecone: {e}")

class RetrievalError(Exception):
    """Exception raised for errors in the Retriever."""
    pass