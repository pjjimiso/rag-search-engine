import argparse

from .search_utils import DEFAULT_ALPHA, DEFAULT_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize keyword and semantic search scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform a search using both keyword and semantic search types")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, help="Weight used to control search type influence (default 0.5)")
    weighted_search_parser.add_argument("--limit", type=int, help="Number of results (default 5)")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)

        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)

        case _:
            parser.print_help()



def normalize_command(scores: list[int]) -> None: 
    if not scores or len(scores) == 0:
        return

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score: 
        for _ in range(len(scores)):
            print("1.0")
        return

    for score in scores: 
        print(round((score - min_score) / (max_score - min_score), 4))


def weighted_search_command(query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_LIMIT) -> None:
    return


if __name__ == "__main__":
    main()
