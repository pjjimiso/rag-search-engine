import argparse

from cli.lib.search_utils import BM25_K1
from lib.keyword_search import (
    search_title,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build index cache")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Count frequency of a term in a specified document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count")

    idf_parser = subparsers.add_parser("idf", help="Calculate IDF for a term")
    idf_parser.add_argument("term", type=str, help="Term to calculate IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate IDF for a term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to calculate IDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Calculate BM25 IDF for a term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to calculate BM25 IDF for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Calculate BM25 TF for a term in a specified document")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to calculate BM25 TF for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="BM25 K1 parameter (default: 1.5)")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
            pass
        case "search":
            print("Searching for:", args.query)
            matches = search_title(args.query)
            for i, match in enumerate(matches, start=1):
                print(f"{i}. ({match['id']}) {match['title']}")
            pass
        case "tf":
            frequency = tf_command(args.doc_id, args.term)
            print(f"Frequency of term '{args.term}' in document {args.doc_id}: {frequency}")
            pass
        case "idf": 
            idf = idf_command(args.term)
            print(f"Inverse document frequency for term '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score for term '{args.term}' in document {args.doc_id}: {tfidf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score for term '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25idf = bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score for term '{args.term}' in document {args.doc_id}: {bm25idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
