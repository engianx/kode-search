import argparse

def main():
    parser = argparse.ArgumentParser(
        prog='kcs',
        description='Build semantic search index for your code base.')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('--prefix', type=str, default='.kode_search', help='Prefix for the various intermidiate files.')

    # arguments for source code extraction
    parser.add_argument('-p', '--parse', action='store_true', default=False, help='Parse code base and extract functions.')
    parser.add_argument('--repo-path', type=str, default='.', help='Path to the code base.')
    parser.add_argument('--include-comments', action='store_true', default=True, help='Include comments in the parsed entities.')

    # arguments for creating embeddings
    parser.add_argument('-e', '--embed', action='store_true', default=False, help='Create embeddings for the code.')
    # arguments for controlling summarizations by LLM
    parser.add_argument('--generate-summary', action='store_true', default=False, help='Generate summary for the code.')
    parser.add_argument('--code-lines-threshold', type=int, default=10, help='Threshold to be considered for summarization.')
    parser.add_argument('--max-summary-words', type=int, default=50, help='(Soft) maximum words in the summary.')
    # arguments for controlling embedding creation
    parser.add_argument('--embedding-file-name', type=str, default='.kode_search_embeddings', help='Name of the embedding file.')
    parser.add_argument('--embedding-type', choices=['code', 'summary', 'summary_and_code'], default='code', help='Type of the index.')
    parser.add_argument('--embedding-samples', type=int, default=1000, help='Dry-run: number of samples to be used for building index.')
    parser.add_argument('--recreate-embeddings', action='store_true', default=False, help='Recreate embeddings from existing embedding file.')
    # arguments for debugging and testing embeddings
    parser.add_argument('--embedding-info', action='store_true', default=False, help='Show information about the embeddings.')
    parser.add_argument('--show-samples', metavar='N', type=int, default=0, help='Show N samples from the embeddings.')

    # arguments for controlling the index
    parser.add_argument('-i', '--index', action='store_true', default=False, help='Create index for the code base.')

    # arguments for serving
    parser.add_argument('-r', '--run', action='store_true', default=False, help='Run the server.')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on.')

    # arguments for command line searching (testing)
    parser.add_argument('-s', '--search', action='store_true', default=False, help='Search code base.')

    parser.add_argument('query', nargs=argparse.REMAINDER, help='Search query (if any).')

    args = parser.parse_args()

    if args.parse:
        from kode_search.parse import parse
        parse(args)
    elif args.embed:
        from kode_search.embed import embed
        embed(args)
    elif args.index:
        from kode_search.index import index
        index(args)
    elif args.run:
        from kode_search.server import run
        run(args)
    elif args.search:
        from kode_search.search import search
        search(args)
    else:
        parser.print_help()

if __name__ == '__main___':
    main()