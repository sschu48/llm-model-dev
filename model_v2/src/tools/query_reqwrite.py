# query will be rewritten to better express user intention

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, question: str, documents: List[Dict[str, Any]], conversation_history: str) -> str:
        logger.debug(f"Rewriting query for question: {question}")
        logger.debug(f"Number of documents: {len(documents)}")

        doc_texts = [self._extract_text(doc) for doc in documents[:3]]  # Use top 3 documents for context
        context = "\n".join(doc_texts)

        prompt = f"""Conversation history:
{conversation_history}

Original question: {question}

Context from documents:
{context}

Rewrite the question to be more specific based on the conversation history and the context from documents. If the question seems complete and specific already, you can return it as is."""
        
        logger.debug(f"Sending prompt to LLM: {prompt}")
        rewritten_query = self.llm.generate(prompt)
        logger.debug(f"Rewritten query: {rewritten_query}")

        return rewritten_query

    def _extract_text(self, document: Dict[str, Any]) -> str:
        logger.debug(f"Extracting text from document: {document}")
        relevant_fields = ['title', 'content', 'text']
        for field in relevant_fields:
            if field in document and isinstance(document[field], str):
                return document[field]
        
        # If no relevant text field is found, return a string representation of the document
        return str(document)

class QueryRewriteError(Exception):
    """Exception raised for errors in the QueryRewriter."""
    pass