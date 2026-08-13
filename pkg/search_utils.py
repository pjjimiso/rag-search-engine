import string

from nltk.stem import PorterStemmer


def search_title(query, movies, stopwords): 
   clean_query = preprocess_text(query)
   tokenized_query = tokenize_text(clean_query, stopwords)
   matches = []
   for movie in movies:
      clean_title = preprocess_text(movie["title"])
      tokenized_title = tokenize_text(clean_title, stopwords)
      if has_matching_token(tokenized_query, tokenized_title):
         matches.append(movie["title"])
   return matches


def has_matching_token(query_tokens, title_tokens):
   for query_token in query_tokens:
      for title_token in title_tokens:
         if query_token in title_token:
            return True
   return False


def preprocess_text(text):
   text = text.lower()
   text = text.translate(str.maketrans('', '', string.punctuation))
   return text


def tokenize_text(text, stopwords):
   text = preprocess_text(text)
   tokens = text.split() 
   stemmer = PorterStemmer()
   filtered_tokens = []
   for token in tokens: 
      if token != "" and token not in stopwords:
         filtered_tokens.append(stemmer.stem(token))
   return filtered_tokens

