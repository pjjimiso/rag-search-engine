import argparse
import json
import string

from pathlib import Path
from pkg.search_utils import search_title


def main() -> None:
    MAX_RESULTS = 5

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    movies_path = Path(__file__).parent.parent / "data" / "movies.json"
    stopwords_path = Path(__file__).parent.parent / "data" / "stopwords.txt"

    match args.command:
        case "search":
            print("Searching for:", args.query)

            with open(movies_path, 'r', encoding='utf-8') as movies_file: 
               movie_dict = json.load(movies_file)

            with open(stopwords_path, 'r', encoding='utf-8') as stopwords_file: 
                stopwords = stopwords_file.read().splitlines()

            punc_table = str.maketrans('', '', string.punctuation)
            clean_stopwords = []
            for stopword in stopwords: 
                clean_word = stopword.translate(punc_table).strip()
                clean_stopwords.append(clean_word)

            matches = search_title(args.query, movie_dict["movies"], clean_stopwords)
            for index, match in enumerate(matches[:MAX_RESULTS], start=1):
                print(f"{index}. {match}")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
