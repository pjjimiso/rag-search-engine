import os
import json

from typing import TypedDict


class Movie(TypedDict):
   id: int
   title: str
   description: str


class SearchResult(TypedDict):
   score: float
   title: str
   description: str


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
FREQUENCY_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")
DOC_LENGTHS_PATH = os.path.join(CACHE_PATH, "doc_lengths.pkl")
EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "movie_embeddings.npy")

MAX_RESULTS = 5
BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_CHUNK_SIZE = 5
DEFAULT_OVERLAP = 2


def load_movies() -> list[Movie]:
   with open(DATA_PATH, "r", encoding="utf-8") as f:
      data = json.load(f)
   return data["movies"]


def setup_cache() -> None: 
   if not os.path.exists(CACHE_PATH):
      os.makedirs(CACHE_PATH)

