from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from models.llm import llm

def generate(docs, question):
    """
    Return llm  generated content based off of retrieved documents and provided question
    """
    # prompt 
    prompt = hub.pull("rlm/rag-prompt")

    # chain formatting llm with prompt provided context
    rag_chain = prompt | llm | StrOutputParser()

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return rag_chain.invoke({"context": docs, "question": question})