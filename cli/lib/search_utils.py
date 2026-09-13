import os
import json

from typing import Any, TypedDict


class Movie(TypedDict):
   id: int
   title: str
   description: str


class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
FREQUENCY_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")
DOC_LENGTHS_PATH = os.path.join(CACHE_PATH, "doc_lengths.pkl")
EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_PATH, "chunk_metadata.json")

MAX_RESULTS = 5
BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_CHUNK_SIZE = 4
DEFAULT_OVERLAP = 1

SCORE_PRECISION = 3


def load_movies() -> list[Movie]:
   with open(DATA_PATH, "r", encoding="utf-8") as f:
      data = json.load(f)
   return data["movies"]


def setup_cache() -> None: 
   if not os.path.exists(CACHE_PATH):
      os.makedirs(CACHE_PATH)


def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
   """Create standardized search result

   Args:
     doc_id: Document ID
     title: Document title
     document: Display text (usually short description)
     score: Relevance/similarity score
     **metadata: Additional metadata to include

   Returns:
     Dictionary representation of search result
   """
   return {
      "id": doc_id,
      "title": title,
      "document": document[:100],
      "score": round(score, SCORE_PRECISION),
      "metadata": metadata if metadata else {},
   }

