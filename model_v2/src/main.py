import os
from dotenv import load_dotenv
from models.llm import LLM
from models.embedder import Embedder
from tools.retrieve import Retriever
from tools.grade_documents import Grader
from tools.query_reqwrite import QueryRewriter
from tools.web_search import WebSearcher
from tools.generate import AnswerGenerator
from interfaces.st import run_streamlit_app

def main():
    # Load environment variables
    load_dotenv()

    # Initialize components
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    llm = LLM(api_key=openai_api_key)
    embedder = Embedder()
    retriever = Retriever(embedder=embedder)
    grader = Grader(llm)
    query_rewriter = QueryRewriter(llm)
    web_searcher = WebSearcher()
    answer_generator = AnswerGenerator(llm)

    # Run the Streamlit app
    run_streamlit_app(
        retriever=retriever,
        grader=grader,
        query_rewriter=query_rewriter,
        web_searcher=web_searcher,
        answer_generator=answer_generator
    )

if __name__ == "__main__":
    main()