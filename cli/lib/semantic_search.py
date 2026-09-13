import os
import re
import json
import numpy as np

from sentence_transformers import SentenceTransformer

from .search_utils import (
   CHUNK_EMBEDDINGS_PATH,
   CHUNK_METADATA_PATH,
   DEFAULT_CHUNK_SIZE,
   EMBEDDINGS_PATH,
   DEFAULT_CHUNK_SIZE,
   DEFAULT_OVERLAP,
   Movie,
   SearchResult,
   setup_cache,
   load_movies, 
   format_search_result,
)


MODEL = "all-MiniLM-L6-v2"


class SemanticSearch:
   def __init__(self, model_name: str = MODEL) -> None:
      self.model = SentenceTransformer(model_name)
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


   def search(self, query: str, limit: int) -> list[dict]:
      if self.embeddings is None or len(self.embeddings) == 0:
         raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
      query_embedding = self.generate_embedding(query)
      score_map = []
      for i, doc_embedding in enumerate(self.embeddings):
         score = cosine_similarity(doc_embedding, query_embedding)
         score_map.append((score, self.documents[i]))

      score_map.sort(key=lambda t: t[0], reverse=True)
      results = []
      for entry in score_map[:limit]:
         results.append({
            "score": float(entry[0]),
            "title": entry[1]["title"],
            "description": entry[1]["description"]
         })
      return results


class ChunkedSemanticSearch(SemanticSearch):
   def __init__(self, model_name: str = MODEL) -> None: 
      super().__init__(model_name)
      self.chunk_embeddings: np.ndarray | None = None
      self.chunk_metadata: list[dict] | None = None


   def build_chunk_embeddings(self, documents: list[Movie], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_SIZE) -> np.ndarray:
      self.documents = documents
      all_chunks = []
      chunk_metadata = []
      for doc in documents:
         self.document_map[doc["id"]] = doc
         if doc["description"] is None or doc["description"].strip() == "":
            continue
         semantic_chunks = semantic_chunk(doc["description"], chunk_size, overlap)
         all_chunks.extend(semantic_chunks)
         for i in range(len(semantic_chunks)):
            chunk_metadata.append({
               "movie_idx": doc["id"],
               "chunk_idx": i,
               "total_chunks": len(semantic_chunks)
            })


      self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
      setup_cache()
      np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

      with open(CHUNK_METADATA_PATH, "w") as f:
         json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

      return self.chunk_embeddings


   def load_or_create_chunk_embeddings(self, documents: list[Movie], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_SIZE) -> np.ndarray:
      self.documents = documents
      for doc in documents: 
         self.document_map[doc["id"]] = doc

      if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
         self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
         assert isinstance(self.chunk_embeddings, np.ndarray), "Loaded embeddings are not a numpy array."
         with open(CHUNK_METADATA_PATH, "r") as f:
            metadata = json.load(f)
            self.chunk_metadata = metadata["chunks"]
         return self.chunk_embeddings

      return self.build_chunk_embeddings(documents, chunk_size, overlap)


   def search_chunks(self, query: str, limit: int = 10) -> list[SearchResult]:
      query_embedding = self.generate_embedding(query)

      assert isinstance(self.chunk_embeddings, np.ndarray), "Loaded embeddings are not a numpy array."
      assert isinstance(self.chunk_metadata, list), "Chunk metadata is not a list."

      chunk_scores = []
      for i, chunk_embedding in enumerate(self.chunk_embeddings):
         score = cosine_similarity(chunk_embedding, query_embedding)
         chunk_scores.append({
            "chunk_idx": i,
            "movie_idx": self.chunk_metadata[i]["movie_idx"],
            "score": score
         })
      movie_scores = {}
      for score in chunk_scores:
         movie_idx = score["movie_idx"]
         if movie_idx not in movie_scores or score["score"] > movie_scores[movie_idx]["score"]:
            movie_scores[movie_idx] = score

      sorted_movie_scores = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)[:limit]
      results = []
      for score in sorted_movie_scores:
         results.append(
            format_search_result(
               score["movie_idx"],
               self.document_map[score["movie_idx"]]["title"],
               self.document_map[score["movie_idx"]]["description"],
               score["score"]
            ))

      return results


def verify_model() -> None:
   try:
      search = SemanticSearch()
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


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def chunk_segments(segments: list[str], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
   i = 0 
   chunks = []
   n_segments = len(segments)

   while i < n_segments:
      chunked_segment = segments[i : i + chunk_size]
      if chunks and len(chunked_segment) <= overlap:
         break

      chunks.append(" ".join(chunked_segment))
      i += chunk_size - overlap

   return chunks


def fixed_length_chunk(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
   words = text.split()
   return chunk_segments(words, chunk_size, overlap)


def fixed_chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> None:
   chunks = fixed_length_chunk(text, chunk_size, overlap)
   print(f"Fixed-length chunking {len(text)} characters.")
   for i, chunk in enumerate(chunks):
      print(chunk)


def semantic_chunk(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
   text = text.strip()
   if text == "":
      return []

   sentences = re.split(r"(?<=[.!?])\s+", text)
   if len(sentences) == 1 and not sentences[0].endswith(('!', '?', '.')):
      return [text]

   cleaned_sentences = [s.strip() for s in sentences if s.strip()]
   return chunk_segments(cleaned_sentences, chunk_size, overlap)


def semantic_chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> None:
   chunks = semantic_chunk(text, chunk_size, overlap)
   print(f"Semantically chunking {len(text)} characters.")
   for i, chunk in enumerate(chunks, start=1):
      print(f"{i}. {chunk}")

