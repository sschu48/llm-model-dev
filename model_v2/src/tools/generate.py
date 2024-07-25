# once documents are received, they will be added to prompt context
# llm will generate response for user based off prompt

class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, question, documents, web_results=None):
        context = "\n".join(documents)
        if web_results:
            context += "\nWeb search results: " + "\n".join(web_results)
        prompt = f"Question: {question}\nContext: {context}\nAnswer the question based on the given context."
        return self.llm.generate(prompt)