import argparse

from lib.keyword_search import search_title

def main() -> None:
    MAX_RESULTS = 5

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)

            matches = search_title(args.query)
            for index, match in enumerate(matches[:MAX_RESULTS], start=1):
                print(f"{index}. {match}")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
