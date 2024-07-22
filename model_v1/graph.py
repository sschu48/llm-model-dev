from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict
from nodes.retrieve_node import retrieve
from nodes.grade_documents_node import grade_documents_node
from nodes.generate_node import generate
from nodes.transform_query_node import transform_query
from nodes.web_search_node import web_search
from nodes.decide_to_generate import decide_to_generate
from nodes.conversation_history_node import manage_conversation_history



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        conversation_history: summarized conversation history
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    conversation_history: str

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents_node)  # grade documents
workflow.add_node("manage_history", manage_conversation_history) # summarize chat history
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search_node", web_search)  # web search

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "manage_history": "manage_history",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "manage_history")
workflow.add_edge("manage_history", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()