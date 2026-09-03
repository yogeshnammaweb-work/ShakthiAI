from fastembed import TextEmbedding


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    def __init__(self):
        self.model = TextEmbedding(
            model_name=MODEL_NAME
        )

    def embed(self, text: str):
        return list(self.model.embed([text]))[0]