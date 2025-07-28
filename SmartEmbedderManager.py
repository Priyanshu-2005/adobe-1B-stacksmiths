import spacy
import numpy as np
import gc
from typing import List, Dict, Optional


class SmartEmbedderManager:
    _nlp: Optional[spacy.language.Language] = None
    _cache: Dict[int, np.ndarray] = {}
    _max_cache_size: int = 1_000

    @classmethod
    def _load_nlp(cls) -> spacy.language.Language:
        if cls._nlp is None:
            cls._nlp = spacy.load(
                "en_core_web_md",
                disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"],
            )
        return cls._nlp

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    @classmethod
    def encode_with_cache(cls, texts: List[str]) -> np.ndarray:
        nlp_model = cls._load_nlp()

        embedding_results: Dict[int, np.ndarray] = {}
        texts_to_process: List[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cache_key = hash(text[:200])
            if cache_key in cls._cache:
                embedding_results[i] = cls._cache[cache_key]
            else:
                texts_to_process.append((i, text))

        if texts_to_process:
            docs = list(
                nlp_model.pipe((text for _, text in texts_to_process), batch_size=128, n_process=1)
            )
            for (original_index, original_text), doc in zip(texts_to_process, docs):
                embedding_vector = cls._unit(doc.vector.astype(np.float32))
                embedding_results[original_index] = embedding_vector
                cls._cache[hash(original_text[:200])] = embedding_vector

            if len(cls._cache) > cls._max_cache_size:
                num_excess_items = len(cls._cache) - cls._max_cache_size
                for _ in range(num_excess_items):
                    cls._cache.pop(next(iter(cls._cache)))

        return np.vstack([embedding_results[i] for i in range(len(texts))])

    @classmethod
    def cleanup(cls) -> None:
        cls._nlp = None
        cls._cache.clear()
        gc.collect()


