import argparse

from lib.keyword_search import search_title, build_command

def main() -> None:
    MAX_RESULTS = 5

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build index cache")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)

            matches = search_title(args.query)
            for index, match in enumerate(matches[:MAX_RESULTS], start=1):
                print(f"{index}. {match}")
            pass
        case "build":
            print("Building index cache...")

            index = build_command()
            docs = index.get_documents('merida')
            print(f"First document for token 'merida' = {docs[0]}")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
