import argparse
import gzip

def main():
    parser = argparse.ArgumentParser(
        prog='kcs',
        description='Build semantic search index for your code base.')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('--verbose', metavar='N', type=int, default=0, help='Verbosity level (0-3).')
    parser.add_argument('--prefix', type=str, default='.kode_search', help='Prefix for the various intermediate files.')
    parser.add_argument('--zip', action='store_true', default=False, help='Zip the intermidiate files.')
    parser.add_argument('--info', action='store_true', default=False, help='Show information of the intermediate files.')
    parser.add_argument('--show-samples', metavar='N', type=int, default=0, help='Show N samples from the intermediate files.')    

    # arguments for source code extraction
    parser.add_argument('-p', '--parse', action='store_true', default=False, help='Parse code base and extract functions.')
    parser.add_argument('--repo-path', type=str, default='.', help='Path to the code base.')
    parser.add_argument('--include-comments', action='store_true', default=True, help='Include comments in the parsed entities.')
    # sampling is done at the parsing stage, so we don't need to check it in following steps.
    # 0 means not sampling, and the default value is 0.
    parser.add_argument('--sampling', type=int, default=0, help='Dry-run: sample a number of class and function only.')

    # Summarization is considered as a part of the parsing process. Although it is an independent step.
    # arguments for controlling summarizations by LLM
    parser.add_argument('--generate-summary', action='store_true', default=False, help='Generate summary for the code.')
    parser.add_argument('--openai-api-retries', type=int, default=3, help='Number of retries for OpenAI API calls.')
    parser.add_argument('--code-lines-threshold', type=int, default=10, help='Threshold to be considered for summarization.')
    parser.add_argument('--max-summary-words', type=int, default=100, help='(Soft) maximum words in the summary.')

    # arguments for creating embeddings
    parser.add_argument('-e', '--embed', action='store_true', default=False, help='Create embeddings for the code.')
    parser.add_argument('--model', metavar='MODEL', type=str, default='all-mpnet-base-v2', help='Embedding model to be used.')
    parser.add_argument('--embedding-type', choices=['code', 'summary', 'summary_and_code'], default='code', help='Type of the index.')
    parser.add_argument('--embedding-batch-size', type=int, default=32, help='Batch size for generating embeddings.')

    # arguments for controlling the index
    parser.add_argument('-i', '--index', action='store_true', default=False, help='Create index for the code base.')
    parser.add_argument('--index-type', choices=['faiss', 'annoy'], default='faiss', help='Type of the index.')
    parser.add_argument('--annoy-num-trees', type=int, default=100, help='Number of trees for annoy index.')

    # arguments for serving
    parser.add_argument('-r', '--run', action='store_true', default=False, help='Run the server.')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on.')

    # arguments for command line searching (testing)
    parser.add_argument('-s', '--search', action='store_true', default=False, help='Search code base.')
    parser.add_argument('--distance-threshold', type=float, default=0.5, help='Similarity distance threshold for search. It may have different meaning for different indexes.')

    parser.add_argument('query', nargs=argparse.REMAINDER, help='Search query (if any).')

    args = parser.parse_args()

    # Set the open function
    if args.zip:
        args._open = gzip.open
    else:
        args._open = open

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