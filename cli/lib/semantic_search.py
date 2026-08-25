from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"

class SemanticSearch:
   def __init__(self) -> None:
      self.model = SentenceTransformer("all-MiniLM-L6-v2")


def verify_model() -> None:
   try:
      search = SemanticSearch()
      # TODO - .model doesn't exist anymore?
      #print(f"Model loaded: {search.model.model}")
      print(f"Max sequence length: {search.model.max_seq_length}")
   except Exception as e:
      print(f"Error loading model: {e}")

