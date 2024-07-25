# retrieve relevant documents from Pinecone
# embed user query and perform similarity search between query and vector db
# return relevant documents

import pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os

class Retriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = "rag-retrieval-v2"
        self._initialize_pinecone()

    def _initialize_pinecone(self):
        try:
            pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
            if self.index_name not in pinecone.list_indexes():
                raise ValueError(f"Index '{self.index_name}' does not exist in Pinecone")
            self.index = pinecone.Index(self.index_name)
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            self.index = None

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def add_documents(self, documents: List[Dict[str, Any]]):
        if not self.index:
            print("Pinecone index not initialized. Cannot add documents.")
            return

        try:
            vectors = []
            for doc in documents:
                embedding = self.encode([doc['text']])[0]
                vector = {
                    'id': doc['id'],
                    'values': embedding,
                    'metadata': {
                        'title': doc.get('title', ''),
                        'page': doc.get('page', ''),
                        'source': doc.get('source', '')
                    }
                }
                vectors.append(vector)
            
            self.index.upsert(vectors=vectors)
            print(f"Successfully added {len(vectors)} documents to Pinecone")
        except Exception as e:
            print(f"Error adding documents to Pinecone: {e}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.index:
            print("Pinecone index not initialized. Cannot retrieve documents.")
            return []

        try:
            query_embedding = self.encode([query])[0]
            results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            
            retrieved_docs = []
            for match in results['matches']:
                doc = {
                    'id': match['id'],
                    'score': match['score'],
                    'title': match['metadata'].get('title', ''),
                    'page': match['metadata'].get('page', ''),
                    'source': match['metadata'].get('source', '')
                }
                retrieved_docs.append(doc)
            
            return retrieved_docs
        except Exception as e:
            print(f"Error retrieving documents from Pinecone: {e}")
            return []

    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        if not self.index:
            print("Pinecone index not initialized. Cannot retrieve metadata.")
            return {}

        try:
            result = self.index.fetch(ids=[doc_id])
            if doc_id in result['vectors']:
                metadata = result['vectors'][doc_id]['metadata']
                return {
                    'title': metadata.get('title', ''),
                    'page': metadata.get('page', ''),
                    'source': metadata.get('source', '')
                }
            else:
                print(f"Document with id '{doc_id}' not found in the index")
                return {}
        except Exception as e:
            print(f"Error retrieving document metadata from Pinecone: {e}")
            return {}
        
class RetrievalError(Exception):
    """Exception raised for errors in the Retriever."""
    pass