from tools.retriever import retriever

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever().get_relevant_documents(question)
    doc_txt = documents[1].page_content
    return {"documents": doc_txt, "question": question}