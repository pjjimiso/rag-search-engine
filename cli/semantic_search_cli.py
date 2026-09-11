import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text
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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
