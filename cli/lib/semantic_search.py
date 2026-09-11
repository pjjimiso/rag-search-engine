import os
import numpy as np
from typing import cast

from sentence_transformers import SentenceTransformer

from .search_utils import EMBEDDINGS_PATH, Movie, setup_cache, load_movies


MODEL = "all-MiniLM-L6-v2"


class SemanticSearch:
   def __init__(self) -> None:
      self.model = SentenceTransformer("all-MiniLM-L6-v2")
      self.embeddings: np.ndarray | None = None 
      self.documents: list[Movie] = []
      self.document_map: dict[int, Movie] = {}


   def generate_embedding(self, text: str) -> np.ndarray:
      if text is None or text.strip() == "":
         raise ValueError("Input text cannot be None or empty.")
      embeddings = self.model.encode([text])
      return embeddings[0]


   def build_embeddings(self, documents: list[Movie]) -> np.ndarray:
      self.documents = documents
      doc_strings = []
      for doc in documents: 
         self.document_map[doc["id"]] = doc
         doc_strings.append(f"{doc['title']}: {doc['description']}")
      self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
      setup_cache()
      np.save(EMBEDDINGS_PATH, self.embeddings)
      return self.embeddings


   def load_or_create_embeddings(self, documents: list[Movie]) -> np.ndarray:
      self.documents = documents
      for doc in documents: 
         self.document_map[doc["id"]] = doc

      if os.path.exists(EMBEDDINGS_PATH):
         self.embeddings = np.load(EMBEDDINGS_PATH)
         assert isinstance(self.embeddings, np.ndarray), "Loaded embeddings are not a numpy array."
         if len(self.embeddings) == len(documents): 
            return self.embeddings

      return self.build_embeddings(documents)


def verify_model() -> None:
   try:
      search = SemanticSearch()
      # TODO - .model doesn't exist anymore?
      #print(f"Model loaded: {search.model.model}")
      print(f"Max sequence length: {search.model.max_seq_length}")
   except Exception as e:
      print(f"Error loading model: {e}")


def embed_text(text: str) -> None:
   s = SemanticSearch()
   embedding = s.generate_embedding(text)
   print(f"Text: {text}")
   print(f"First 3 dimensions: {embedding[:3]}")
   print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None: 
   s = SemanticSearch()
   embedding = s.generate_embedding(query)
   print(f"Query: {query}")
   print(f"First 3 dimensions: {embedding[:3]}")
   print(f"Shape: {embedding.shape}")


def verify_embeddings() -> None: 
   s = SemanticSearch()
   documents = load_movies()
   embeddings = s.load_or_create_embeddings(documents)
   print(f"Number of docs:   {len(documents)}")
   print(
       f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
   )


