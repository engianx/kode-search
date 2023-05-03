import fnmatch
import git
import openai
import os
import pickle
import random
import sys
import tiktoken
import time

from collections import Counter
from datetime import datetime
from pprint import pprint
from tree_sitter_languages import get_parser
from tqdm import tqdm

from kode_search.constants import FILE_EXTENSIONS
from kode_search.utils import ask_user_confirmation, num_tokens_from_messages

SUPPORTED_FILE_EXTENSIONS = {
    '.c': 'c',
    '.h': 'cpp', # TODO: be smart about C or C++ header files
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

ENTITIES_TO_EXTRACT = [
    'class_definition', 
    'class_declaration', # Java class declaration
    'function_definition',
    'method_definition', 
    'function_declaration', 
    'method_declaration',
    'linkage_specification',  # for C++ JNI calls.
]

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

def _is_class_node_type(node_type):
    return node_type == 'class_definition' or node_type == 'class_declaration'

def _is_function_node_type(node_type):
    return (node_type == 'function_definition' 
            or node_type == 'method_definition'
            or node_type == 'function_declaration'
            or node_type == 'method_declaration'
            or node_type == 'linkage_specification')

# Extract functions and classes from the source code
def _extract_entities(args, tree, source_file, source_code):
    entities = []
    nodes_to_visit = list(tree.root_node.children)
    for node in nodes_to_visit:
        # Only extract functions and classes at the root level
        if node.type not in ENTITIES_TO_EXTRACT:
            if args.verbose >= 3:
                print('Skip node {} in file {}'.format(node, source_file))
            continue

        # We can have comments right before the function/class
        # We want to keep the comments as part of the function/class
        start_node = node
        # Include comments if specified
        if args.include_comments:
            if node.prev_sibling is not None and node.prev_sibling.type == 'comment':
                comment = node.prev_sibling
                # only if the comment is close to the function/class
                if comment.end_point[0] == node.start_point[0] - 1:
                    if args.verbose >= 3:
                        print('Insert comment node {} before {}'.format(comment, node))
                    start_node = comment

        entities.append({
            'type': node.type,
            'file': source_file,
            'start_line': start_node.start_point[0],
            'end_line': node.end_point[0],
            'code': source_code[start_node.start_byte:node.end_byte],
        })
        # Recursively visit classes.
        if _is_class_node_type(node.type):
            nodes_to_visit.extend(node.children)
    return entities

# show the information of the parsed entities
def _show_info(args, entities_file):
    if not os.path.exists(entities_file):
        print('File {} does not exist.'.format(entities_file))
        sys.exit(1)

    with args._open(entities_file, 'rb') as f:
        dataset = pickle.load(f)

    print('repo:\t {}'.format(dataset['repo']))
    print('created at:\t{}'.format(dataset['created_at']))

    entities = dataset['entities']
    print('Number of entities:\t{}'.format(len(entities)))

    # Print the number classes and functions
    num_class = len([entity for entity in entities if _is_class_node_type(entity['type'])])
    num_function = (len([entity for entity in entities if _is_function_node_type(entity['type'])]))
    print('\tclasses:\t{}'.format(num_class))
    print('\tfunctions:\t{}'.format(num_function))
    
    # Bucket the code length into different ranges
    bucket_size = 1000
    max_code_length = max([len(entity['code']) for entity in entities])
    buckets = [0] * (max_code_length // bucket_size + 1)
    for entity in entities:
        buckets[len(entity['code']) // bucket_size] += 1
    print('Code length distribution:')
    print('Length\tCount')
    for i, count in enumerate(buckets):
        if count == 0:
            continue
        print('{}-{}\t\t{}'.format(i * bucket_size, (i+1)*bucket_size, count))

# Show some samples of the parsed entities
def _show_samples(args, entities_file):
    if not os.path.exists(entities_file):
        print('File {} does not exist.'.format(entities_file))
        sys.exit(1)

    with args._open(entities_file, 'rb') as f:
        dataset = pickle.load(f)
    
    entities = dataset['entities']
    for entity in random.sample(entities, min(args.show_samples, len(entities))):
        pprint(entity, width=120)
        print()

# GPT api parameters
GPT_MODEL = 'gpt-3.5-turbo'
GPT_MAX_TOKENS = 2048
GPT_TEMPERATURE = 0.8
GPT_TOP_P = 0.5
GPT_FREQUENCY_PENALTY = 0
GPT_PRESENCE_PENALTY = 0

# call openai chat completion api with back-off
# returns a json string if successful, otherwise None
def _get_code_summary(args, code): 
    # If the code is too short, we don't need to summarize it
    if code.count('\n') < args.code_lines_threshold:
        if args.verbose >= 2:
            print('Code is too short to summarize.')
        return ""
    
    # GPT chat completion api prompt.
    prompt_template = """Summarize the the code. Suggest a few keywords from the code. Use less than {} words in total.

    ```
    {}
    ```
    """
    
    prompt = prompt_template.format(args.max_summary_words, code)
    messages = [{"role": "user", "content": prompt}]

    # check if the message is too long
    num_tokens = num_tokens_from_messages(messages)
    while num_tokens > GPT_MAX_TOKENS:
        if args.verbose >= 2:
            print('Code is too long, removing some lines. {} tokens > {} tokens'.format(num_tokens, GPT_MAX_TOKENS))
        # estimate how many lines we should remove
        percent = GPT_MAX_TOKENS / num_tokens
        splitted_code_lines = code.split('\n')
        num_lines = int(len(splitted_code_lines) * percent)
        code = '\n'.join(splitted_code_lines[:num_lines])

        prompt = prompt_template.format(args.max_summary_words, code)
        messages = [{"role": "user", "content": prompt}]
        num_tokens = num_tokens_from_messages(messages)

    # if the final message is too small, we don't need to summarize it
    if code.count('\n') < args.code_lines_threshold:
        if args.verbose >= 2:
            print('Code is too short to summarize.')
        return ""

    try_count = 0
    while try_count < args.openai_api_retries:
        try_count += 1
        try:
            response = openai.ChatCompletion.create(
                messages=messages,
                model=GPT_MODEL,
                temperature=GPT_TEMPERATURE,
                max_tokens=GPT_MAX_TOKENS,
                top_p=GPT_TOP_P,
                frequency_penalty=GPT_FREQUENCY_PENALTY,
                presence_penalty=GPT_PRESENCE_PENALTY
            )
            return response.choices[0].message["content"]
        except Exception as e:
            if args.verbose >= 1:
                print('GPT API Error: {}'.format(e))
            time.sleep(2**try_count)

    if args.verbose >= 1:
        print('Failed to get summary.')
    return ""

# Generate summaries for the parsed entities
# This step costs $$$.
def _generate_summaries(args, entities_file):
    print('Generating summaries for entities in {}...'.format(entities_file))
    if not ask_user_confirmation("This step costs $$$, are you sure to continue?"):
        sys.exit(0)

    if not os.path.exists(entities_file):
        print('File {} does not exist.'.format(entities_file))
        sys.exit(1)

    with args._open(entities_file, 'rb') as f:
        dataset = pickle.load(f)

    entities = dataset['entities']
    for entity in tqdm(entities):
        summary = _get_code_summary(args, entity['code'])
        entity['summary'] = summary

    with args._open(entities_file, 'wb') as f:
        pickle.dump(dataset, f)

# Parse functions and classes from the source code, and save them in a file.
def _parse_source_code(args, output_file):
    source_files = _get_source_files(args.repo_path)
    if len(source_files) == 0:
        print('No supported source code files found in {}.'.format(args.repo_path))
        sys.exit(1)
    
    if args.verbose >= 1:
        print('Found {} source code files.'.format(len(source_files)))
    if args.verbose >= 2:
        print('Source code file samples:')
        for file in random.sample(source_files, 10):
            print('\t{}'.format(file))

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

# Parse functions and classes from the source code, and save them in a file.
def parse(args):
    entities_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['parse'])

    if args.info:
        _show_info(args, entities_file)
        return
    
    if args.show_samples > 0:
        _show_samples(args, entities_file)
        return

    if args.generate_summary:
        _generate_summaries(args, entities_file)
        return
    
    print("parsing source code in {} ...".format(args.repo_path))

    _parse_source_code(args, entities_file)

    print("parsed entities saved to {}".format(entities_file))
