import string
import pickle
import math

from collections import Counter
from nltk.stem import PorterStemmer

from .search_utils import (
   STOPWORDS_PATH,
   INDEX_PATH,
   DOCMAP_PATH,
   FREQUENCY_PATH,
   DOC_LENGTHS_PATH,
   MAX_RESULTS,
   BM25_K1,
   BM25_B,
   Movie,
   load_movies,
   setup_cache
)


class InvertedIndex: 
   def __init__(self) -> None: 
      self.index: dict[str, set[int]] = {}
      self.docmap: dict[int, Movie] = {}
      self.term_frequencies: dict[int, Counter] = {}
      self.doc_lengths: dict[int, int] = {}

   def __add_document(self, doc_id: int, text: str) -> None:
      tokens = tokenize_text(text)
      token_count = 0
      for token in tokens: 
         self.index.setdefault(token, set()).add(doc_id)
         self.term_frequencies.setdefault(doc_id, Counter())[token] += 1
         token_count += 1
      self.doc_lengths[doc_id] = token_count

   def __get_avg_doc_length(self) -> float:
      if len(self.doc_lengths) == 0:
         return 0.0
      return sum(self.doc_lengths.values()) / len(self.doc_lengths)

   def get_documents(self, term: str) -> list[int]:
      if term not in self.index: 
         return []
      return sorted(self.index[term])

   def get_tf(self, doc_id: int, token: str) -> int:
      return self.term_frequencies.get(doc_id, {}).get(token, 0)

   def get_idf(self, token: str) -> float:
      return math.log(
         (len(self.docmap) + 1) / (len(self.get_documents(token)) + 1)
      )

   def get_tfidf(self, doc_id: int, token: str) -> float:
      return self.get_tf(doc_id, token) * self.get_idf(token)

   def build(self) -> None: 
      movies = load_movies()
      for m in movies: 
         self.__add_document(m["id"], f"{m['title']} {m['description']}")
         self.docmap[m["id"]] = m

   def save(self) -> None:
      setup_cache()
      with open(INDEX_PATH, "wb") as index_file:
         pickle.dump(self.index, index_file)
      with open(DOCMAP_PATH, "wb") as docmap_file:
         pickle.dump(self.docmap, docmap_file)
      with open(FREQUENCY_PATH, "wb") as frequency_file:
         pickle.dump(self.term_frequencies, frequency_file)
      with open(DOC_LENGTHS_PATH, "wb") as doclengths_file:
         pickle.dump(self.doc_lengths, doclengths_file)

   def load(self) -> None: 
      with open(INDEX_PATH, "rb") as index_file: 
         self.index = pickle.load(index_file)
      with open(DOCMAP_PATH, "rb") as docmap_file: 
         self.docmap = pickle.load(docmap_file)
      with open(FREQUENCY_PATH, "rb") as frequency_file: 
         self.term_frequencies = pickle.load(frequency_file)
      with open(DOC_LENGTHS_PATH, "rb") as doclengths_file: 
         self.doc_lengths = pickle.load(doclengths_file)

   def get_bm25_idf(self, term: str) -> float:
      n = len(self.docmap)
      df = len(self.get_documents(tokenize_single_term(term)))
      return math.log((n - df + 0.5) / (df + 0.5) + 1)
   
   def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
      if self.__get_avg_doc_length() == 0:
         length_norm = 0.0
      else:
         length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
      tf = self.get_tf(doc_id, tokenize_single_term(term))
      return (tf * (k1 + 1) / (tf + k1 * length_norm))

   def bm25(self, doc_id: int, term: str) -> float:
      bm25_idf = self.get_bm25_idf(term)
      bm25_tf = self.get_bm25_tf(doc_id, term)
      return bm25_idf * bm25_tf


def build_command() -> None:
   index = InvertedIndex()
   index.build()
   index.save()
   return


def search_title(query: str) -> list[dict]: 
   index = InvertedIndex()
   index.load() 
   clean_query = preprocess_text(query)
   tokenized_query = tokenize_text(clean_query)
   matching_indices = []
   for token in tokenized_query: 
      matching_indices.extend(index.get_documents(token)[:MAX_RESULTS])
      if len(matching_indices) >= 5:
         break
   return matching_indices


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
   for query_token in query_tokens:
      for title_token in title_tokens:
         if query_token in title_token:
            return True
   return False


def preprocess_text(text: str) -> str:
   text = text.lower()
   text = text.translate(str.maketrans('', '', string.punctuation))
   return text


def load_stopwords() -> list[str]:
   with open(STOPWORDS_PATH, "r") as f: 
      return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
   text = preprocess_text(text)
   tokens = text.split() 
   stemmer = PorterStemmer()
   final_tokens = []
   for token in tokens: 
      if token != "" and token not in STOPWORDS:
         final_tokens.append(stemmer.stem(token))
   return final_tokens


def tokenize_single_term(term: str) -> str: 
   token = tokenize_text(term)
   if len(token) > 1: 
      raise ValueError(f"Expected a single token, but got {len(token)} tokens.")
   return token[0]


def tf_command(doc_id: int, term: str) -> int: 
   index = InvertedIndex()
   index.load()
   return index.get_tf(doc_id, tokenize_single_term(term))


def idf_command(term: str) -> float:
   index = InvertedIndex()
   index.load()
   return index.get_idf(tokenize_single_term(term))


def tfidf_command(doc_id: int, term: str) -> float:
   index = InvertedIndex()
   index.load()
   return index.get_tfidf(doc_id, tokenize_single_term(term))


def bm25_idf_command(term: str) -> float:
   index = InvertedIndex()
   index.load()
   return index.get_bm25_idf(tokenize_single_term(term))


def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1) -> float:
   index = InvertedIndex()
   index.load()
   return index.get_bm25_tf(doc_id, tokenize_single_term(term), k1)

