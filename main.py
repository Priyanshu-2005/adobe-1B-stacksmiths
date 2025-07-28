from __future__ import annotations
import json, sys, datetime, re, string, heapq, gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import pdfplumber
import spacy
from scipy.spatial.distance import cdist
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

import json, sys, datetime, re, string, heapq, gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, Counter

import numpy as np
import pdfplumber
import networkx as nx
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class EmbeddingHandler:
    _spacy_model: Optional[spacy.language.Language] = None
    _vector_cache: Dict[int, np.ndarray] = {}
    _cache_limit: int = 1_000

    @classmethod
    def _get_model(cls) -> spacy.language.Language:
        if cls._spacy_model is None:
            cls._spacy_model = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])
        return cls._spacy_model

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    @classmethod
    def get_embeddings(cls, text_list: List[str]) -> np.ndarray:
        nlp_instance = cls._get_model()
        encoded_vectors: Dict[int, np.ndarray] = {}
        pending_encoding: List[Tuple[int, str]] = []

        for i, text_item in enumerate(text_list):
            text_hash = hash(text_item[:200])
            if text_hash in cls._vector_cache:
                encoded_vectors[i] = cls._vector_cache[text_hash]
            else:
                pending_encoding.append((i, text_item))

        if pending_encoding:
            spacy_docs = list(nlp_instance.pipe((t for _, t in pending_encoding), batch_size=128))
            for (i, _), spacy_doc in zip(pending_encoding, spacy_docs):
                vector = cls._normalize_vector(spacy_doc.vector.astype(np.float32))
                encoded_vectors[i] = vector
                cls._vector_cache[hash(text_list[i][:200])] = vector

            if len(cls._vector_cache) > cls._cache_limit:
                num_to_evict = len(cls._vector_cache) - cls._cache_limit
                for _ in range(num_to_evict):
                    cls._vector_cache.pop(next(iter(cls._vector_cache)))

        return np.vstack([encoded_vectors[i] for i in range(len(text_list))])

    @classmethod
    def clear_resources(cls):
        cls._spacy_model = None
        cls._vector_cache.clear()
        gc.collect()


@dataclass
class DocumentSegment:
    source_file: str
    page_number: int
    segment_id: int
    content: str
    start_index: int
    end_index: int
    similarity_score: float = 0.0
    keyword_match_score: float = 0.0
    final_score: float = 0.0


@dataclass
class DocumentPage:
    source_file: str
    page_number: int
    content: str
    similarity_score: float = 0.0
    segments: List[DocumentSegment] = None

    def __post_init__(self):
        if self.segments is None:
            self.segments = []


def get_relevant_keywords(input_text: str, count: int = 15) -> List[str]:
    cleaned_text = re.sub(r"[^\w\s]", " ", input_text.lower())
    word_list = cleaned_text.split()

    domain_specific_terms = {
        "business": ["strategy", "market", "revenue", "profit", "customer", "growth", "sales"],
        "technical": ["system", "process", "method", "algorithm", "data", "analysis", "performance"],
        "legal": ["compliance", "regulation", "policy", "requirement", "standard", "audit"],
        "finance": ["cost", "budget", "investment", "return", "financial", "economic"],
    }

    frequency_counter = Counter(word_list)
    common_words = {
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    }
    high_freq_words = [
        w for w, _ in frequency_counter.most_common(50) if len(w) > 2 and w not in common_words
    ]

    domain_related_words = []
    for _, terms in domain_specific_terms.items():
        domain_related_words.extend([t for t in terms if t in cleaned_text])

    all_keywords = list(dict.fromkeys(high_freq_words + domain_related_words))
    return all_keywords[:count]


def get_keyphrases(input_text: str, count: int = 10) -> List[str]:
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(2, 4),
            max_features=50,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform([input_text])
        features = vectorizer.get_feature_names_out()
        score_values = matrix.toarray()[0]
        ranked_phrases = sorted(zip(features, score_values), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, score in ranked_phrases[:count] if score > 0]
    except Exception:
        return []


def segment_page_content(
    page_text: str,
    source_file: str,
    page_index: int,
    segment_size: int = 400,
    overlap_size: int = 50,
) -> List[DocumentSegment]:
    sentence_list = re.split(r"(?<=[.!?])\s+", page_text)
    segments: List[DocumentSegment] = []

    current_segment_text = ""
    start_position = 0
    segment_counter = 0

    for sentence in sentence_list:
        sentence = sentence.strip()
        if not sentence:
            continue

        potential_segment = f"{current_segment_text} {sentence}" if current_segment_text else sentence
        if len(potential_segment.split()) > segment_size and current_segment_text:
            segments.append(
                DocumentSegment(
                    source_file=source_file,
                    page_number=page_index,
                    segment_id=segment_counter,
                    content=current_segment_text.strip(),
                    start_index=start_position,
                    end_index=start_position + len(current_segment_text),
                )
            )
            overlap_content = " ".join(current_segment_text.split()[-overlap_size:])
            current_segment_text = f"{overlap_content} {sentence}" if overlap_content else sentence
            start_position += len(current_segment_text) - len(f"{overlap_content} {sentence}")
            segment_counter += 1
        else:
            current_segment_text = potential_segment

    if current_segment_text.strip():
        segments.append(
            DocumentSegment(
                source_file=source_file,
                page_number=page_index,
                segment_id=segment_counter,
                content=current_segment_text.strip(),
                start_index=start_position,
                end_index=start_position + len(current_segment_text),
            )
        )
    return segments


def calculate_hybrid_scores(segments: List[DocumentSegment], user_persona: str, user_task: str) -> List[DocumentSegment]:
    if not segments:
        return segments

    search_query = f"{user_persona}. {user_task}"
    search_keywords = set(get_relevant_keywords(search_query))
    search_phrases = set(get_keyphrases(search_query))

    texts_to_embed = [search_query] + [c.content for c in segments]
    vector_embeddings = EmbeddingHandler.get_embeddings(texts_to_embed)

    query_vector = vector_embeddings[0]
    segment_vectors = vector_embeddings[1:]
    similarity_scores = segment_vectors @ query_vector

    for i, segment in enumerate(segments):
        lower_text = segment.content.lower()
        word_set = set(lower_text.split())

        keyword_overlap_score = len(search_keywords & word_set) / max(len(search_keywords), 1)
        phrase_match_count = sum(1 for p in search_phrases if p.lower() in lower_text)
        phrase_match_score = phrase_match_count / max(len(search_phrases), 1)
        content_length_score = min(len(segment.content.split()) / 200, 1.0)

        segment.similarity_score = float(similarity_scores[i])
        segment.keyword_match_score = keyword_overlap_score * 0.4 + phrase_match_score * 0.6
        segment.final_score = 0.5 * similarity_scores[i] + 0.3 * segment.keyword_match_score + 0.2 * content_length_score

    return segments


def filter_pages_by_relevance(pages_list: List[DocumentPage], user_persona: str, user_task: str, page_limit: int = 80) -> List[DocumentPage]:
    if len(pages_list) <= page_limit:
        return pages_list

    query_keywords = set(get_relevant_keywords(f"{user_persona} {user_task}"))
    query_phrases = set(get_keyphrases(f"{user_persona} {user_task}"))

    scored_pages: List[Tuple[DocumentPage, float]] = []

    for page_item in pages_list:
        lower_text = page_item.content.lower()
        word_set = set(lower_text.split())

        keyword_density = len(query_keywords & word_set) / max(len(query_keywords), 1)
        phrase_matches = sum(1 for p in query_phrases if p.lower() in lower_text)
        text_richness = min(len(page_item.content.split()) / 500, 1.0)

        indicator_words = [
            "summary", "conclusion", "key", "important", "critical",
            "main", "primary", "executive", "overview", "findings",
        ]
        indicator_bonus = sum(0.1 for word in indicator_words if word in lower_text)
        quick_score = 0.4 * keyword_density + 0.3 * (phrase_matches / max(len(query_phrases), 1)) + 0.2 * text_richness + 0.1 * min(indicator_bonus, 0.5)

        scored_pages.append((page_item, quick_score))

    scored_pages.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored_pages[:page_limit]]


def extract_pdf_pages(pdf_file_path: Path, min_word_count: int = 30) -> List[Tuple[int, str]]:
    page_data: List[Tuple[int, str]] = []
    try:
        with pdfplumber.open(pdf_file_path) as doc:
            for page_num, page_obj in enumerate(doc.pages, 1):
                page_text = None
                try:
                    page_text = page_obj.extract_text(layout=True, x_tolerance=2, y_tolerance=2)
                except Exception:
                    pass
                if not page_text:
                    try:
                        page_text = page_obj.extract_text()
                    except Exception:
                        continue
                if not page_text:
                    continue

                text_lines = page_text.split("\n")
                if len(text_lines) > 4:
                    if len(text_lines[0].split()) < 5 and any(c.isdigit() for c in text_lines[0]):
                        text_lines = text_lines[1:]
                    if len(text_lines) > 0 and len(text_lines[-1].split()) < 5 and any(c.isdigit() for c in text_lines[-1]):
                        text_lines = text_lines[:-1]

                page_text = "\n".join(text_lines)
                page_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", page_text)
                page_text = re.sub(r"[ \t]+", " ", page_text)
                page_text = re.sub(r"-\n", "", page_text).strip()

                word_list = page_text.split()
                if len(word_list) >= min_word_count:
                    alpha_char_ratio = sum(1 for w in word_list[:20] if w.isalpha()) / min(len(word_list), 20)
                    if alpha_char_ratio > 0.5:
                        page_data.append((page_num, page_text))
    except Exception as e:
        print(f"Warning: error processing {pdf_file_path}: {e}")
    return page_data


def summarize_with_textrank(input_text: str, num_sentences: int = 5) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", input_text) if s.strip()]
    if len(sentences) <= num_sentences:
        return input_text

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1, 2))
        sentence_vectors = vectorizer.fit_transform(sentences)

        if len(sentences) > num_sentences * 3:
            num_clusters = max(num_sentences, min(num_sentences * 2, len(sentences) // 3))
            kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_ids = kmeans_model.fit_predict(sentence_vectors)
            sentences_by_cluster: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
            for i, cid in enumerate(cluster_ids):
                sentences_by_cluster[cid].append((i, sentences[i]))

            selected_sentences: List[str] = []
            for cid, sentence_list in sentences_by_cluster.items():
                if len(selected_sentences) >= num_sentences:
                    break
                indices = [i for i, _ in sentence_list]
                vectors = sentence_vectors[indices]
                similarity_matrix = (vectors @ vectors.T).toarray()
                centrality_scores = np.mean(similarity_matrix, axis=1)
                best_index = max(range(len(indices)), key=lambda j: centrality_scores[j])
                selected_sentences.append(sentence_list[best_index][1])

            if len(selected_sentences) < num_sentences:
                remaining_indices = set(range(len(sentences))) - {
                    i for sents in sentences_by_cluster.values() for i, _ in sents
                }
                residual_sentences = [(np.sum(sentence_vectors[i].toarray()), sentences[i]) for i in remaining_indices]
                residual_sentences.sort(reverse=True)
                selected_sentences.extend([s for _, s in residual_sentences[: num_sentences - len(selected_sentences)]])

            return " ".join(selected_sentences[:num_sentences])

        similarity_matrix = (sentence_vectors @ sentence_vectors.T).toarray()
        np.fill_diagonal(similarity_matrix, 0)
        graph = nx.from_numpy_array(similarity_matrix)
        pagerank_scores = nx.pagerank(graph, alpha=0.85, max_iter=50, tol=1e-4)
        ranked_indices = sorted(range(len(sentences)), key=lambda i: pagerank_scores[i], reverse=True)
        return " ".join(sentences[i] for i in ranked_indices[:num_sentences])

    except Exception:
        return " ".join(sentences[:num_sentences])


def _trim_text(text: str, word_count: int) -> str:
    return " ".join(text.split()[:word_count])


def _default_headline(text: str, word_count: int) -> str:
    first_sentence = re.split(r"[.!?]", text, 1)[0]
    return _trim_text(first_sentence, word_count).title()


def generate_contextual_headline(
    input_text: str,
    user_persona: str,
    user_task: str,
    max_ngram_size: int = 4,
    top_n: int = 3,
    max_word_count: int = 12,
) -> str:
    text_sample = input_text[:1500]
    try:
        candidate_phrases: Set[str] = set()

        vectorizer = CountVectorizer(ngram_range=(2, max_ngram_size), stop_words="english", max_features=100)
        candidate_phrases.update(vectorizer.fit([text_sample]).get_feature_names_out())

        candidate_phrases.update(get_keyphrases(text_sample, count=20))

        for sentence in re.split(r"(?<=[.!?])\s+", text_sample)[:3]:
            words = sentence.split()
            if 4 <= len(words) <= max_word_count:
                candidate_phrases.add(sentence.strip())

        if not candidate_phrases:
            return _default_headline(text_sample, max_word_count)

        all_texts_to_embed = [text_sample, f"{user_persona}. {user_task}"] + list(candidate_phrases)
        embeddings = EmbeddingHandler.get_embeddings(all_texts_to_embed)

        page_vector, query_vector = embeddings[0], embeddings[1]
        candidate_vectors = embeddings[2:]

        page_similarity = candidate_vectors @ page_vector
        query_similarity = candidate_vectors @ query_vector
        length_scores = np.array([min(len(c.split()) / 8, 1.0) for c in candidate_phrases])
        position_scores = np.array([max(0, 1 - text_sample.lower().find(c.lower()) / 1000) for c in candidate_phrases])

        final_scores = 0.4 * page_similarity + 0.3 * query_similarity + 0.2 * length_scores + 0.1 * position_scores
        candidate_list = list(candidate_phrases)
        ordered_indices = np.argsort(-final_scores)

        chosen_phrases: List[str] = []
        used_words: Set[str] = set()
        for index in ordered_indices:
            phrase = candidate_list[index]
            words = set(phrase.lower().split())
            if len(words & used_words) < len(words) * 0.6:
                used_words |= words
                chosen_phrases.append(phrase)
                if len(chosen_phrases) == top_n:
                    break

        if not chosen_phrases:
            return _default_headline(text_sample, max_word_count)

        main_headline = _trim_text(chosen_phrases[0], max_word_count).title()
        if len(chosen_phrases) == 1:
            return main_headline
        sub_headlines = [_trim_text(p, max_word_count // 2).title() for p in chosen_phrases[1:]]
        if len(sub_headlines) == 1:
            return f"{main_headline}: {sub_headlines[0]}"
        return f"{main_headline}: {', '.join(sub_headlines[:-1])} and {sub_headlines[-1]}"

    except Exception:
        return _default_headline(text_sample, max_word_count)


def run_processing_pipeline(config_data: Dict, config_file_path: str) -> Dict:
    config_dir = Path(config_file_path).parent.resolve()

    user_persona = config_data["persona"]["role"]
    user_task = config_data["job_to_be_done"]["task"]

    print(f"Persona: {user_persona}")
    print(f"Task:    {user_task}")

    pages_list: List[DocumentPage] = []

    document_files = [(config_dir / d["filename"]).resolve() for d in config_data["documents"]]

    print(f"Reading {len(document_files)} PDFs…")
    for file_path in tqdm(document_files, desc="PDF", ncols=80):
        for page_num, page_text in extract_pdf_pages(file_path):
            pages_list.append(DocumentPage(str(file_path), page_num, page_text))
    if not pages_list:
        raise ValueError("No pages extracted.")

    pages_list = filter_pages_by_relevance(pages_list, user_persona, user_task, page_limit=80)

    all_segments: List[DocumentSegment] = []
    for page_item in tqdm(pages_list, desc="Chunk", ncols=80):
        segments = segment_page_content(page_item.content, page_item.source_file, page_item.page_number)
        page_item.segments = segments
        all_segments.extend(segments)

    scored_segments = calculate_hybrid_scores(all_segments, user_persona, user_task)

    document_buckets: Dict[str, List[DocumentSegment]] = defaultdict(list)
    for segment in scored_segments:
        document_buckets[segment.source_file].append(segment)

    document_scores: List[Tuple[str, float, DocumentSegment]] = []
    for file_name, segment_list in document_buckets.items():
        top_5_segments = heapq.nlargest(5, segment_list, key=lambda c: c.final_score)
        average_score = float(np.mean([c.final_score for c in top_5_segments]))
        best_segment = max(top_5_segments, key=lambda c: c.final_score)
        document_scores.append((file_name, average_score, best_segment))

    document_scores.sort(key=lambda x: -x[1])
    top_documents = document_scores[:5]

    extracted_data = []
    analysis_data = []

    for rank, (file_name, _, best_segment) in enumerate(top_documents, 1):
        page_segments = sorted(
            [c for c in document_buckets[file_name] if c.page_number == best_segment.page_number], key=lambda c: c.segment_id
        )
        full_page_text = " ".join(c.content for c in page_segments)

        headline = generate_contextual_headline(full_page_text, user_persona, user_task)
        summary = summarize_with_textrank(full_page_text, 7)

        extracted_data.append(
            {
                "document": Path(file_name).name,
                "section_title": headline,
                "importance_rank": rank,
                "page_number": best_segment.page_number,
            }
        )
        analysis_data.append(
            {
                "document": Path(file_name).name,
                "refined_text": summary,
                "page_number": best_segment.page_number,
            }
        )

    return {
        "metadata": {
            "input_documents": [d["filename"] for d in config_data["documents"]],
            "persona": user_persona,
            "job_to_be_done": user_task,
            "processing_timestamp": datetime.datetime.utcnow().isoformat(),
        },
        "extracted_sections": extracted_data,
        "subsection_analysis": analysis_data,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python persona_di_intel.py input.json output.json")
        sys.exit(1)

    input_path, output_path = sys.argv[1:]

    try:
        with open(input_path, encoding="utf-8") as f:
            config = json.load(f)

        output_result = run_processing_pipeline(config, input_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_result, f, indent=4, ensure_ascii=False)

        print(f"✓ Results written to {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        EmbeddingHandler.clear_resources()
