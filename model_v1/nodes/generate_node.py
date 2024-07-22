from tools.generate import rag_chain

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    conversation_history = state["conversation_history"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain().invoke({"context": documents, "question": conversation_history})
    return {"documents": documents, "question": question, "generation": generation, "conversation_history": conversation_history}