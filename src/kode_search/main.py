import argparse
import logging
import gzip

def main():
    parser = argparse.ArgumentParser(
        prog='kcs',
        description='Build semantic search index for your code base.')

    # Common sub-commands
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO',
                        help='Set the logging level (default: %(default)).')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Set the log file.')   
    parser.add_argument('--prefix', type=str, default='.kode_search',
                        help='Prefix for the various intermediate files.')
    parser.add_argument('--zip', action='store_true', default=True,
                        help='Zip the intermidiate files.')
    parser.add_argument('--info', action='store_true', default=False,
                        help='Show information of the intermediate files.')
    parser.add_argument('--show-samples', metavar='N', type=int, default=0,
                        help='Show N samples from the intermediate files.')    
    # By default, we don't automatically run the command, unless --run is specified.
    parser.add_argument('--run', action='store_true', default=False,
                        help='Run the command.')

    # arguments for source code extraction
    parser.add_argument('-p', '--parse', action='store_true', default=False,
                        help='Parse code base and extract functions.')
    parser.add_argument('--repo-path', type=str, default='.',
                        help='Path to the code base.')
    # sampling is done at the parsing stage, so we don't need to check it in following steps.
    # 0 means not sampling, and the default value is 0.
    parser.add_argument('--sampling', type=int, default=0,
                        help='Dry-run: sample a number of class and function only.')

    # Summarization is considered as a part of the parsing process. Although it is an independent step.
    # arguments for controlling summarizations by LLM
    parser.add_argument('--generate-summary', action='store_true',
                        default=False, help='Generate summary for the code.')
    parser.add_argument('--openai-api-retries', type=int, default=3,
                        help='Number of retries for OpenAI API calls.')
    parser.add_argument('--code-lines-threshold', type=int, default=10,
                        help='Threshold to be considered for summarization.')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads to be used for summarization.')

    # arguments for creating embeddings
    parser.add_argument('-e', '--embed', action='store_true', default=False,
                        help='Create embeddings for the code.')
    parser.add_argument('--model', metavar='MODEL', type=str, default='all-mpnet-base-v2',
                        help='Embedding model to be used.')
    parser.add_argument('--embedding-type', choices=['summary', 'code', 'summary_and_code'], default='summary',
                        help='Type of the embedding.')
    parser.add_argument('--embedding-batch-size', type=int, default=32,
                        help='Batch size for generating embeddings.')

    # arguments for controlling the index
    parser.add_argument('-i', '--index', action='store_true', default=False,
                        help='Create index for the code base.')
    parser.add_argument('--index-type', choices=['faiss', 'annoy'], default='faiss',
                        help='Type of the index.')
    parser.add_argument('--annoy-num-trees', type=int, default=100,
                        help='Number of trees for annoy index.')

    # arguments for command line searching (testing)
    parser.add_argument('-s', '--search', action='store_true', default=False,
                        help='Search code base.')
    parser.add_argument('--distance-threshold', type=float, default=0.5,
                        help='Similarity distance threshold for search. It may have different meaning for different indexes.')

    # arguments for serving
    parser.add_argument('-x', '--server', action='store_true', default=False,
                        help='Start the server.')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on.')
    parser.add_argument('--file-url-template', type=str, default='https://cs.android.com/android/platform/superproject/+/master:art/{file};l={line}',
                        help='URL template for the link to result file path.')

    # arguments for searching
    parser.add_argument('query', nargs=argparse.REMAINDER, metavar='ARGS',
                        help='Search query (if any).')

    args = parser.parse_args()

    # set logging information
    logging.basicConfig(level=args.log_level, filename=args.log_file,
                        format='%(asctime)s %(levelname)s %(message)s')

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
    elif args.server:
        from kode_search.server import run
        run(args)
    elif args.search:
        from kode_search.search import search
        search(args)
    else:
        parser.print_help()

if __name__ == '__main___':
    main()