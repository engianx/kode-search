import fnmatch
import git
import logging
import os
import pickle
import random
import shutil
import sys

from datetime import datetime
from tree_sitter_languages import get_parser
from tqdm import tqdm

from kode_search.constants import FILE_EXTENSIONS

SUPPORTED_FILE_EXTENSIONS = {
    '.c': 'c',
    '.h': 'cpp', # TODO: be smart about C or C++ header files
    '.cc': 'cpp',
    '.cpp': 'cpp',
    '.hpp': 'cpp',
    '.java': 'java',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.py': 'python',
    '.go': 'go',
}

EXCLUDED_FILE_PATTERNS = [
    '*test*',
    '*Test*',
    '*/.*', # hidden files
]

# Language specific node types to extract
# Note: comment type can be single line or multiple line depending on the language
_LANG_TO_NODE_TYPES = {
    'java': ['class_declaration', 'interface_declaration', 'method_declaration'],
    'cpp': ['class_specifier', 'struct_specifier', 'function_definition'],
    'c': ['function_definition', 'linkage_specification'],
    'python': ['class_definition', 'function_definition'],
    'javascript': ['function_declaration'],
    'typescript': ['function_declaration'],
    'go': ['function_declaration'],
}

NODE_TYPES_TO_EXTRACT = list(set(val for sublit in _LANG_TO_NODE_TYPES.values() for val in sublit))

def _get_language(source_file):
    return SUPPORTED_FILE_EXTENSIONS[os.path.splitext(source_file)[1]]

def _get_source_files_from_git_repo(repo_path):
    try:
        repo = git.Repo(repo_path, search_parent_directories=False)
        # Get source code files of supported types from the git repo
        source_code_files = []
        for file in repo.git.ls_files().split('\n'):
            if file.endswith(tuple(SUPPORTED_FILE_EXTENSIONS.keys())):
                source_code_files.append(os.path.join(repo_path, file))
        return source_code_files
    except git.exc.InvalidGitRepositoryError:
        return None
    
def _get_source_files(repo_path):
    # First check if it is a git repo
    source_files = _get_source_files_from_git_repo(repo_path)
    if source_files is None:
        # Walk through the directory to find all source code files
        source_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(tuple(SUPPORTED_FILE_EXTENSIONS.keys())):
                    source_files.append(os.path.join(root, file))
    # Filter out files matching the excluded file patterns
    source_files = [file for file in source_files if not any([fnmatch.fnmatch(file, pattern) for pattern in EXCLUDED_FILE_PATTERNS])]
    return source_files

def _traverse_tree(tree):
    cursor = tree.walk()
    reached_root = False
    while reached_root is False:
        yield cursor.node
        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            if cursor.goto_next_sibling():
                retracing = False

# Customization code
def _filter_node(node, source_file, source_code):
    lang = _get_language(source_file)
    return node.start_point[0] == node.end_point[0]

# Cap the content length to 32KB, roughly 8K tokens, that's text-embedding-ada-002's limit.
CONTENT_LENGTH_CAP = 32 * 1024 # 32KB
# Extract functions and classes from the source code
def _extract_entities(tree, source_file, source_code):
    entities = []
    all_nodes = list(_traverse_tree(tree))
    for node in all_nodes:
        # Only extract functions and classes at the root level
        if node.type not in NODE_TYPES_TO_EXTRACT:
            continue

        if _filter_node(node, source_file, source_code):
            continue

        # Entity format:
        # {
        #   'file': 'path/to/file.java',
        #   'start_line': 10,
        #   'end_line': 20,
        #   'content_type': 'code', or 'text'
        #   'content': 'class Foo { ... }', can also be comments, etc.
        # }
        # In the summary generation step, it will add 'summary' field to the entity for 'code' type text.

        # Cap the content length to 16KB, roughly 4K tokens
        content_end = min(node.end_byte, node.start_byte + CONTENT_LENGTH_CAP)

        entities.append({
            'file': source_file,
            'start_line': node.start_point[0],
            'end_line': node.end_point[0],
            'content_type': 'code', # or 'text'
            'content': source_code[node.start_byte:content_end],
        })
    return entities

# Parse functions and classes from the source code, and save them in a file.
def _parse_source_code(args, output_file):
    logging.info('Prasing source code from repo...')
    source_files = _get_source_files(args.repo_path)
    if len(source_files) == 0:
        logging.warning('No supported source code files found in {}.'.format(args.repo_path))
        return
    
    logging.info('Found {} source code files.'.format(len(source_files)))
    logging.debug('Source code file samples:')
    for file in random.sample(source_files, 10):
        logging.debug('\t{}'.format(file))

    entities = []
    # Load the parser for the language, and parse the source code
    for source_file in tqdm(source_files):
        try:
            with open(source_file, 'r') as f:
                source_code = f.read()
            parser = get_parser(_get_language(source_file))
            tree = parser.parse(bytes(source_code, 'utf8'))
        except UnicodeDecodeError:
            logging.error('UnicodeDecodeError when parsing {}'.format(source_file))
            continue
        entities.extend(_extract_entities(tree, source_file, source_code))

    if args.sampling > 0:
        entities = random.sample(entities, min(args.sampling, len(entities)))
        print('Sampled {} entities.'.format(len(entities)))

    dataset = {
        'repo': args.repo_path,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'entities': entities,
    }
    
    with args._open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    logging.info("Parsed entities saved to {}".format(output_file))

# Recreate the entities file from the source code.
def _recreate(args, entities_file):
    logging.info('Recreating entities file {}...'.format(entities_file))
    if not os.path.exists(entities_file):
        logging.critical('File {} does not exist.'.format(entities_file))
        sys.exit(1)

    # Make a backup of the original file
    backup_file = entities_file + '.bak'
    shutil.copyfile(entities_file, backup_file)
    logging.info('Backup file saved to {}'.format(backup_file))

    with args._open(entities_file, 'rb') as f:
        dataset = pickle.load(f)

    entities = dataset['entities']

    # Remove oneliners
    new_entities = [e for e in entities if len(e['content'].split('\n')) > 1]
    logging.info('Removed {} oneliners.'.format(len(entities) - len(new_entities)))
    dataset['entities'] = new_entities    

    with args._open(entities_file, 'wb') as f:
        pickle.dump(dataset, f)

# Both function are exposed to summary.py
# show the information of the parsed entities
def show_info(args, entities_file):
    if not os.path.exists(entities_file):
        logging.critical('File {} does not exist.'.format(entities_file))
        sys.exit(1)

    with args._open(entities_file, 'rb') as f:
        dataset = pickle.load(f)

    print('repo:\t {}'.format(dataset['repo']))
    print('created at:\t{}'.format(dataset['created_at']))

    entities = dataset['entities']
    print('Number of entities:\t{}'.format(len(entities)))
    # Number of entities with summary
    print('Number of summaries:\t{}'.format(sum(1 for e in entities if 'summary' in e)))
    
    # Bucket the code length into different ranges
    bucket_size = 1024
    max_code_length = max([len(entity['content']) for entity in entities])
    buckets = [0] * (max_code_length // bucket_size + 1)
    for entity in entities:
        buckets[len(entity['content']) // bucket_size] += 1
    print('Code length distribution:')
    print('Length\tCount')
    for i, count in enumerate(buckets):
        print('{}-{}\t\t{}'.format(i * bucket_size, (i+1)*bucket_size, count))

# Show some samples of the parsed entities
def show_samples(args, entities_file):
    if not os.path.exists(entities_file):
        logging.critical('File {} does not exist.'.format(entities_file))
        sys.exit(1)

    with args._open(entities_file, 'rb') as f:
        dataset = pickle.load(f)
    
    entities = dataset['entities']

    samples = random.sample(entities, min(args.show_samples, len(entities)))
    from kode_search.viewer import Viewer
    Viewer(samples).run()


# Parse functions and classes from the source code, and save them in a file.
def parse(args):
    entities_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['parse'])

    if args.info:
        show_info(args, entities_file)
        return
    
    if args.show_samples > 0:
        show_samples(args, entities_file)
        return
    
    if args.recreate:
        _recreate(args, entities_file)
        return
    
    if args.run:
        _parse_source_code(args, entities_file)
        return

    print('No supported action specified, use one of the following options:')
    print('\t--info\tshow info about the parsed entities')
    print('\t--show-samples\tshow samples of the parsed entities')
    print('\t--generate-summary\tgenerate summaries for the parsed entities')
    print('\t--recreate\trecreate the parsed entities')
    print('\t--run\tparse source code and save the parsed entities to a file')