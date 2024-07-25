# this is the embedding model to embed the user query when needing to find documentation
# this will match the vector structure of the whats in Pinecone

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts)