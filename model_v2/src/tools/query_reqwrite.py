# query will be rewritten to better express user intention

class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, question, documents):
        context = "\n".join(documents[:3])  # Use top 3 documents for context
        prompt = f"Original question: {question}\nContext: {context}\nRewrite the question to be more specific based on the context."
        return self.llm.generate(prompt)