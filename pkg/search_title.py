
def search_title(query, data): 
   matches = []
   for movie in data["movies"]:
      if query.lower() in movie["title"].lower(): 
         matches.append(movie["title"])

   return matches
