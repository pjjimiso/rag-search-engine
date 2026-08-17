import argparse

from lib.keyword_search import search_title, build_command, InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    
    subparsers.add_parser("build", help="Build index cache")

    args = parser.parse_args()

    movie_index = InvertedIndex()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            try:
                matches = search_title(args.query, movie_index)
                for i, match in enumerate(matches, start=1):
                    print(f"{i}. {movie_index.docmap[match]['title']}")
            except FileNotFoundError as error: 
                print(f"Failed to get index data: {error}")
            pass
        case "build":
            print("Building index cache...")

            build_command(movie_index)
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
