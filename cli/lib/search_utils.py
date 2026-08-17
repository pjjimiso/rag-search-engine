import os
import json

from typing import TypedDict


class Movie(TypedDict):
   id: int
   title: str
   description: str


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
DOCMAP_PATH = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")


def load_movies() -> list[Movie]:
   with open(DATA_PATH, "r", encoding="utf-8") as f:
      data = json.load(f)
   return data["movies"]


def setup_cache() -> None: 
   if not os.path.exists(CACHE_PATH):
      os.makedirs(CACHE_PATH)

