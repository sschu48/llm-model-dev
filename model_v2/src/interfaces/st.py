# how we will communicate with streamlit
# this might be in the main file. who knows

import streamlit as st
from typing import List, Dict, Any

def run_streamlit_app(retriever, grader, query_rewriter, web_searcher, answer_generator):
    st.title("Corrective RAG Model")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Enter your question:")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        try:
            with st.spinner("Processing your question..."):
                # Step 1: Retrieve relevant documents
                retrieved_docs = retriever.retrieve(question)
                if not retrieved_docs:
                    st.warning("No relevant documents were found. The answer may be less accurate.")

                # Step 2: Grade the retrieved documents
                grade_result = grader.grade(question, retrieved_docs)

                # Step 3: Decide on the path based on grading
                if grade_result == "No":
                    # Direct path to Answer Generation
                    answer = answer_generator.generate(question, retrieved_docs)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    # Step 4: Rewrite the query
                    rewritten_query = query_rewriter.rewrite(question, retrieved_docs)

                    # Step 5: Perform web search
                    web_results = web_searcher.get_search_context(rewritten_query)
                    if not web_results:
                        st.warning("Web search didn't return any results. The answer may be less up-to-date.")

                    # Step 6: Generate answer
                    answer = answer_generator.generate(question, retrieved_docs, web_results)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                # Display the answer
                st.subheader("Answer:")
                st.write(answer)

            # Display intermediate results for debugging
            with st.expander("See processing details"):
                st.write("Retrieved Documents:")
                for doc in retrieved_docs:
                    st.write(f"- Title: {doc['title']}, Source: {doc['source']}, Score: {doc['score']}")
                st.write(f"Grade Result: {grade_result}")
                if grade_result == "Yes":
                    st.write(f"Rewritten Query: {rewritten_query}")
                    st.write("Web Search Results:")
                    st.text(web_results)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": "I'm sorry, but an error occurred while processing your question. Please try again or contact support if the problem persists."})

    # Display chat history
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write("You: " + message["content"])
        else:
            st.write("Assistant: " + message["content"])

def display_doc_details(doc: Dict[str, Any]):
    st.write(f"Title: {doc['title']}")
    st.write(f"Source: {doc['source']}")
    st.write(f"Page: {doc['page']}")
    st.write(f"Relevance Score: {doc['score']}")