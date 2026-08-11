import argparse
import json

from pathlib import Path
from pkg.search_title import search_title


def main() -> None:
    MAX_RESULTS = 5

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    movies_path = Path(__file__).parent.parent / "data" / "movies.json"

    match args.command:
        case "search":
            print("Searching for:", args.query)

            with open(movies_path, 'r', encoding='utf-8') as file: 
               data = json.load(file)

            matches = search_title(args.query, data)
            for index, match in enumerate(matches[:MAX_RESULTS], start=1):
                print(f"{index}. {match}")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
