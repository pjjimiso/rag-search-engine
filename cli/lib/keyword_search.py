import string
import pickle

from nltk.stem import PorterStemmer

from .search_utils import STOPWORDS_PATH, INDEX_PATH, DOCMAP_PATH, MAX_RESULTS, Movie, load_movies, setup_cache


class InvertedIndex: 
   def __init__(self) -> None: 
      self.index: dict[str, set[int]] = {}
      self.docmap: dict[int, Movie] = {}

   def __add_document(self, doc_id: int, text: str) -> None:
      tokens = tokenize_text(text)
      for token in tokens: 
         self.index.setdefault(token, set()).add(doc_id)

   def get_documents(self, term: str) -> list[int]:
      if term not in self.index: 
         return []
      return sorted(self.index[term])

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

   def load(self) -> None: 
      with open(INDEX_PATH, "rb") as index_file: 
         self.index = pickle.load(index_file)
      with open(DOCMAP_PATH, "rb") as docmap_file: 
         self.docmap = pickle.load(docmap_file)


def build_command(index: InvertedIndex) -> None:
   index.build()
   index.save()
   return


def search_title(query: str, index: InvertedIndex) -> list[int]: 
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

