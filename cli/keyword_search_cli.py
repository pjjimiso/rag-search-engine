import argparse

from lib.keyword_search import search_title, build_command, get_frequency, InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build index cache")

    tf_parser = subparsers.add_parser("tf", help="Count frequency of a term in a specified document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count")

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
            frequency = get_frequency(args.doc_id, args.term)
            print(f"Frequency of term '{args.term}' in document {args.doc_id}: {frequency}")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
