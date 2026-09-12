import argparse

from lib.search_utils import (
    load_movies,
    MAX_RESULTS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP
)

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    fixed_chunk_text,
    semantic_chunk_text,
    SemanticSearch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify_model", help="Verify model loading")
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    text_embedding = subparsers.add_parser("embed_text", help="Generate embedding for the provided text")
    text_embedding.add_argument("text", type=str, help="Text to embed")

    query_embedding = subparsers.add_parser("embed_query", help="Generate embedding for the provided query")
    query_embedding.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=MAX_RESULTS, help="Limit the number of results (default: 5)")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk the provided text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Number of words per chunk (default: 5)")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Number of overlapping words in each chunk (default: 2)")

    semantic_chunk = subparsers.add_parser("semantic_chunk", help="Chunk the provided text into smaller pieces")
    semantic_chunk.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk.add_argument("--max-chunk-size", type=int, default=4, help="Number of sentences per semantic chunk (default: 4)")
    semantic_chunk.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences in each chunk (default: 0)")

    args = parser.parse_args()

    match args.command:
        case "verify_model":
           verify_model()

        case "verify_embeddings":
            verify_embeddings()

        case "embed_text":
            embed_text(args.text)

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            search_command(args.query, args.limit)

        case "chunk": 
            chunk_command(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk": 
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)

        case _:
            parser.print_help()


def search_command(query: str, limit: int = MAX_RESULTS) -> None: 
    search = SemanticSearch()
    movies = load_movies()
    search.load_or_create_embeddings(movies)
    results = search.search(query, limit)
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description'][:150]} ...\n")


def chunk_command(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> None:
    fixed_chunk_text(text, chunk_size, overlap)
    #words = text.split()
    #text_chunks = [(" ".join(words[:chunk_size]))]
    #words = words[chunk_size:]
    #while words:
    #    chunk = " ".join(words[:chunk_size])
    #    if overlap > 0:
    #        chunk_overlap = " ".join(text_chunks[-1].split()[-overlap:])  # Keep the last `overlap` words from the previous chunk
    #        text_chunks.append(f"{chunk_overlap} {chunk}")
    #    else:
    #        text_chunks.append(chunk)
    #    words = words[chunk_size:]

    #for i, chunk in enumerate(text_chunks, start=1):
    #    print(f"{i}. {chunk}")

def semantic_chunk_command(text: str, max_chunk_size: int = 4, overlap: int = 0) -> None:
    semantic_chunk_text(text, max_chunk_size, overlap)
    #for i, chunk in enumerate(chunks, start=1):
    #   print(f"{i}. {chunk}")


if __name__ == "__main__":
    main()

