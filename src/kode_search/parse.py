import fnmatch
import git
import os
import pickle
import random
import sys

from collections import Counter
from pprint import pprint
from tree_sitter_languages import get_parser
from tqdm import tqdm

from kode_search.constants import FILE_EXTENSIONS

SUPPORTED_FILE_EXTENSIONS = {
    '.c': 'c',
    '.h': 'c',
    '.cc': 'cpp',
    '.cpp': 'cpp',
    '.hpp': 'cpp',
    '.go': 'go',
    '.java': 'java',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.py': 'python',
    '.rb': 'ruby',
    '.rs': 'rust',
    '.scala': 'scala',
    '.swift': 'swift',
    '.php': 'php',
}

EXCLUDED_FILE_PATTERNS = [
    '*_test.h',
    '*_test.cc',
    '*_test.cpp',
    '*_test.hpp',
    '*_test.go',
    '*Test.java',
    '*_test.js',
    '*_test.ts',
    '*test.py',
    '*/test/*',
    '*/tests/*',
    '*/testing/*',
    '*/__tests__/*',
]

ENTITIES_TO_EXTRACT = ['class_declartion', 'function_definition', '', 'method_definition']

def _get_source_files_from_git_repo(repo_path):
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
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

# Extract functions and classes from the source code
def _extract_entities(args, tree, source_file, source_code):
    entities = []
    nodes_to_visit = list(tree.root_node.children)
    for node in nodes_to_visit:
        # Only extract functions and classes at the root level
        if node.type not in ENTITIES_TO_EXTRACT:
            continue

        # We can have comments right before the function/class
        # We want to keep the comments as part of the function/class
        start_node = node
        # Include comments if specified
        if args.include_comments:
            if node.prev_sibling is not None and node.prev_sibling.type == 'comment':
                comment = node.prev_sibling
                # only if the comment is close to the function/class
                if comment.end_point[0] == node.start_point[0] - 2:
                    start_node = comment

        entities.append({
            'type': node.type,
            'file': source_file,
            'start_line': start_node.start_point[0],
            'end_line': node.end_point[0],
            'code': source_code[start_node.start_byte:node.end_byte],
        })
        # Recursively visit classes.
        if node.type == 'class_declaration':
            nodes_to_visit.extend(node.children)
    return entities

# show the information of the parsed entities
def _show_info(args, entities_file):
    with args._open(entities_file, 'rb') as f:
        entities = pickle.load(f)
    print('Number of entities: {}'.format(len(entities)))

    # Print the number classes and functions
    num_class = len([entity for entity in entities if entity['type'] == 'class_declaration'])
    num_function = len([entity for entity in entities if entity['type'] == 'function_definition'])
    print('\tclasses: {}'.format(num_class))
    print('\tfunctions: {}'.format(num_function))
    
    print("Code length distribution:")
    code_length = [len(entity['code']) for entity in entities]
    # print the distribution of the code length
    print('Length\tCount')
    for value, count in Counter(code_length).most_common(10):
        print('{}\t{}'.format(value, count))

# Show some samples of the parsed entities
def _show_samples(args, entities_file):
    with args._open(entities_file, 'rb') as f:
        entities = pickle.load(f)
    for entity in random.sample(entities, min(args.show_samples, len(entities))):
        pprint(entity, width=120)
        print()

# Parse functions and classes from the source code, and save them in a file.
def _parse_source_code(args, output_file):
    source_files = _get_source_files(args.repo_path)
    if len(source_files) == 0:
        print('No supported source code files found in {}.'.format(args.repo_path))
        sys.exit(1)
    
    entities = []
    # Load the parser for the language, and parse the source code
    for source_file in tqdm(source_files):
        with open(source_file, 'r') as f:
            source_code = f.read()
        parser = get_parser(SUPPORTED_FILE_EXTENSIONS[os.path.splitext(source_file)[1]])
        try:
            tree = parser.parse(bytes(source_code, 'utf8'))
        except UnicodeDecodeError:
            print('UnicodeDecodeError when parsing {}'.format(source_file))
            continue
        entities.extend(_extract_entities(args, tree, source_file, source_code))
    
    with args._open(output_file, 'wb') as f:
        pickle.dump(entities, f)

# Parse functions and classes from the source code, and save them in a file.
def parse(args):
    entities_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['parse'])

    if args.info:
        _show_info(args, entities_file)
        return
    
    if args.show_samples > 0:
        _show_samples(args, entities_file)
        return

    print("parsing source code in {} ...".format(args.repo_path))

    _parse_source_code(args, entities_file)

    print("parsed entities saved to {}".format(entities_file))
