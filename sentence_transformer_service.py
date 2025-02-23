from sentence_transformers import SentenceTransformer, util


class SentenceTransformerService:
    def __init__(self, dataset: dict[str, str]):
        print("Loading sentence transformer model")
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._candidates = list(dataset.keys())
        print("Encoding dataset...")
        self._candidates_embeddings = self._model.encode(self._candidates, convert_to_tensor=True)

    def find_top_n_matches(self, text: str, n: int) -> list[str]:
        text_embedding = self._model.encode(text, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(text_embedding, self._candidates_embeddings).squeeze(0)

        filtered_matches = [
            (i, similarities[i].item()) for i in range(len(self._candidates)) if self._candidates[i] != text
        ]

        filtered_matches.sort(key=lambda x: x[1], reverse=True)

        return [self._candidates[i] for i, _ in filtered_matches[:n]]
