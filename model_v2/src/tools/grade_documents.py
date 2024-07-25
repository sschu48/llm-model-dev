# once documents are retrieved, they will be graded based off their relevance to query
# this will act like a router
# if documents aren't relevant, model will search web

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Grader:
    def __init__(self, llm):
        self.llm = llm

    def grade(self, question: str, documents: List[Dict[str, Any]], conversation_history: str) -> str:
        logger.debug(f"Grading relevance for question: {question}")
        logger.debug(f"Number of documents: {len(documents)}")
        
        # Extract the relevant text from each document
        doc_texts = [self._extract_text(doc) for doc in documents]
        
        # Join the extracted texts
        context = "\n".join(doc_texts)
        
        prompt = f"""Conversation history:
{conversation_history}

Current question: {question}

Context from documents:
{context}

Based on the conversation history, the current question, and the provided context from documents, is the context relevant to answer the question? Answer Yes or No."""

        logger.debug(f"Sending prompt to LLM: {prompt}")
        response = self.llm.generate(prompt)
        logger.debug(f"LLM response: {response}")
        
        result = "Yes" if "yes" in response.lower() else "No"
        logger.debug(f"Grading result: {result}")
        return result

    def _extract_text(self, document: Dict[str, Any]) -> str:
        logger.debug(f"Extracting text from document: {document}")
        relevant_fields = ['title', 'content', 'text']
        for field in relevant_fields:
            if field in document and isinstance(document[field], str):
                return document[field]
        
        # If no relevant text field is found, return a string representation of the document
        return str(document)

class GradingError(Exception):
    """Exception raised for errors in the Grader."""
    pass