import string

from nltk.stem import PorterStemmer
from pickle import dump

from .search_utils import STOPWORDS_PATH, INDEX_PATH, DOCMAP_PATH, Movie, load_movies, setup_cache


class InvertedIndex: 
   def __init__(self) -> None: 
      self.index: dict[str, set[int]] = {}
      self.docmap: dict[int, Movie] = {}

   def __add_document(self, doc_id: int, text: str) -> None:
      tokens = tokenize_text(text)
      for token in tokens: 
         self.index.setdefault(token, set()).add(doc_id)

   def get_documents(self, term: str) -> list[int]:
      return sorted(self.index[term])

   def build(self) -> None: 
      movies = load_movies()
      for m in movies: 
         self.__add_document(m["id"], f"{m['title']} {m['description']}")
         self.docmap[m["id"]] = m

   def save(self) -> None:
      setup_cache()
      with open(INDEX_PATH, "wb") as index_file:
         dump(self.index, index_file)
      with open(DOCMAP_PATH, "wb") as docmap_file:
         dump(self.docmap, docmap_file)


def build_command() -> InvertedIndex:
   index = InvertedIndex()
   index.build()
   index.save()
   return index


def search_title(query: str) -> list[dict]: 
   movies = load_movies()
   clean_query = preprocess_text(query)
   tokenized_query = tokenize_text(clean_query)
   matches = []
   for movie in movies:
      clean_title = preprocess_text(movie["title"])
      tokenized_title = tokenize_text(clean_title)
      if has_matching_token(tokenized_query, tokenized_title):
         matches.append(movie["title"])
   return matches


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

