import string

from nltk.stem import PorterStemmer

from .search_utils import STOPWORDS_PATH, load_movies


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
