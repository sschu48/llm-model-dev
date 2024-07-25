# how we will communicate with streamlit
# this might be in the main file. who knows

import streamlit as st
from typing import List, Dict, Any
import traceback
import logging
from utils.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

def run_streamlit_app(retriever, grader, query_rewriter, web_searcher, answer_generator):
    st.title("Corrective RAG Model with Conversation History")

    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()

    question = st.text_input("Enter your question:")

    if question:
        st.session_state.conversation_manager.add_message("user", question)

        try:
            with st.spinner("Processing your question..."):
                conversation_history = st.session_state.conversation_manager.get_history_as_string()

                logger.debug("Retrieving documents")
                retrieved_docs = retriever.retrieve(question, conversation_history)
                logger.debug(f"Retrieved documents: {retrieved_docs}")
                if not retrieved_docs:
                    st.warning("No relevant documents were found. The answer may be less accurate.")

                logger.debug("Grading documents")
                grade_result = grader.grade(question, retrieved_docs, conversation_history)
                logger.debug(f"Grade result: {grade_result}")

                if grade_result == "No":
                    logger.debug("Generating answer directly")
                    answer = answer_generator.generate(question, retrieved_docs, conversation_history)
                else:
                    logger.debug("Rewriting query")
                    rewritten_query = query_rewriter.rewrite(question, retrieved_docs, conversation_history)
                    logger.debug(f"Rewritten query: {rewritten_query}")

                    logger.debug("Performing web search")
                    web_results = web_searcher.get_search_context(rewritten_query)
                    logger.debug(f"Web search results: {web_results}")
                    if not web_results:
                        st.warning("Web search didn't return any results. The answer may be less up-to-date.")

                    logger.debug("Generating answer with web results")
                    answer = answer_generator.generate(question, retrieved_docs, conversation_history, web_results)

                logger.debug(f"Generated answer: {answer}")

                if not isinstance(answer, str):
                    logger.warning(f"Answer is not a string. Type: {type(answer)}. Converting to string.")
                    answer = str(answer)
                
                st.session_state.conversation_manager.add_message("assistant", answer)

                st.subheader("Answer:")
                st.write(answer)

            with st.expander("See processing details"):
                st.write("Retrieved Documents:")
                for doc in retrieved_docs:
                    st.write(f"- Title: {doc.get('title', 'N/A')}, Source: {doc.get('source', 'N/A')}, Score: {doc.get('score', 'N/A')}")
                st.write(f"Grade Result: {grade_result}")
                if grade_result == "Yes":
                    st.write(f"Rewritten Query: {rewritten_query}")
                    st.write("Web Search Results:")
                    st.text(str(web_results))

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"An error occurred: {str(e)}\n\nStack trace:\n{traceback.format_exc()}")
            st.session_state.conversation_manager.add_message("assistant", "I'm sorry, but an error occurred while processing your question. Please try again or contact support if the problem persists.")

    st.subheader("Conversation History")
    for message in st.session_state.conversation_manager.get_history():
        if message["role"] == "user":
            st.write("You: " + message["content"])
        else:
            st.write("Assistant: " + message["content"])

    if st.button("Clear Conversation History"):
        st.session_state.conversation_manager.clear_history()
        st.experimental_rerun()

def display_doc_details(doc: Dict[str, Any]):
    st.write(f"Title: {doc.get('title', 'N/A')}")
    st.write(f"Source: {doc.get('source', 'N/A')}")
    st.write(f"Page: {doc.get('page', 'N/A')}")
    st.write(f"Relevance Score: {doc.get('score', 'N/A')}")