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

    # Summarize the entities.
    parser.add_argument('-m', '--summary', action='store_true',
                        default=False, help='Generate summary for the code.')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads to be used for summarization.')
    
    parser.add_argument('--recreate', action='store_true', default=False,
                        help='Recreate the entity file from exsiting one.')

    # arguments for creating embeddings
    parser.add_argument('-e', '--embed', action='store_true', default=False,
                        help='Create embeddings for the code.')
    parser.add_argument('--model-name', metavar='MODEL', type=str, default='mpnet',
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
    parser.add_argument('--cuda-device', type=str, default='cuda:0',
                        help='Cuda device to be used for searching.')

    # arguments for serving
    parser.add_argument('-x', '--server', action='store_true', default=False,
                        help='Start the server.')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on.')
    parser.add_argument('--url-tpl', type=str, default=None,
                        help='URL template name for the link to result file path.')
    parser.add_argument('--auto-confirm', action='store_true', default=False,
                        help='Auto confirm the prompt, for auto start the server.')

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
    elif args.summary:
        from kode_search.summary import summarize
        summarize(args)
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