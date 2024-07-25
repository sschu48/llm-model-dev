# once documents are retrieved, they will be graded based off their relevance to query
# this will act like a router
# if documents aren't relevant, model will search web

class Grader:
    def __init__(self, llm):
        self.llm = llm

    def grade(self, question, documents):
        context = "\n".join(documents)
        prompt = f"Question: {question}\nContext: {context}\nIs the context relevant to answer the question? Answer Yes or No."
        response = self.llm.generate(prompt)
        return "Yes" if "yes" in response.lower() else "No"