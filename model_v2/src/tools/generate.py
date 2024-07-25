# once documents are received, they will be added to prompt context
# llm will generate response for user based off prompt

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, question: str, documents: List[Dict[str, Any]], conversation_history: str, web_results: str = None) -> str:
        logger.debug(f"Generating answer for question: {question}")
        logger.debug(f"Number of documents: {len(documents)}")

        doc_texts = [self._extract_text(doc) for doc in documents]
        context = "\n".join(doc_texts)

        if web_results:
            context += f"\nWeb search results: {web_results}"

        prompt = f"""Conversation history:
{conversation_history}

Current question: {question}

Context from documents and web search:
{context}

Based on the conversation history, the current question, and the provided context, generate a comprehensive and accurate answer. Ensure the answer is coherent with the ongoing conversation."""
        
        logger.debug(f"Sending prompt to LLM: {prompt}")
        answer = self.llm.generate(prompt)
        logger.debug(f"Generated answer: {answer}")

        return answer

    def _extract_text(self, document: Dict[str, Any]) -> str:
        logger.debug(f"Extracting text from document: {document}")
        relevant_fields = ['title', 'content', 'text']
        for field in relevant_fields:
            if field in document and isinstance(document[field], str):
                return document[field]
        
        # If no relevant text field is found, return a string representation of the document
        return str(document)

class AnswerGenerationError(Exception):
    """Exception raised for errors in the AnswerGenerator."""
    pass