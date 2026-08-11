import string

def search_title(query, data): 
   matches = []
   for movie in data["movies"]:
      punc_table = str.maketrans('', '', string.punctuation)
      clean_query = query.lower().translate(punc_table)
      clean_title = movie["title"].lower().translate(punc_table)
      tokenized_query = tokenize_string(clean_query)
      if any(word in clean_title for word in tokenized_query):
         matches.append(movie["title"])
   return matches

def tokenize_string(query):
   tokens = query.split() 
   filtered_tokens = [x for x in tokens if x != ""]
   return filtered_tokens
