import fnmatch
import git
import logging
import openai
import os
import pickle
import random
import shutil
import sys
import tiktoken
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
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
    '.java': 'java',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.py': 'python',
    '.go': 'go',
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

# Extract functions and classes from the source code
def _extract_entities(args, tree, source_file, source_code):
    entities = []
    all_nodes = list(_traverse_tree(tree))
    for node in all_nodes:
        # Only extract functions and classes at the root level
        if node.type not in NODE_TYPES_TO_EXTRACT:
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
        entities.append({
            'file': source_file,
            'start_line': node.start_point[0],
            'end_line': node.end_point[0],
            'content_type': 'code', # or 'text'
            'content': source_code[node.start_byte:node.end_byte],
        })
    return entities

# show the information of the parsed entities
def _show_info(args, entities_file):
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
    bucket_size = 1000
    max_code_length = max([len(entity['content']) for entity in entities])
    buckets = [0] * (max_code_length // bucket_size + 1)
    for entity in entities:
        buckets[len(entity['content']) // bucket_size] += 1
    print('Code length distribution:')
    print('Length\tCount')
    for i, count in enumerate(buckets):
        if count == 0:
            continue
        print('{}-{}\t\t{}'.format(i * bucket_size, (i+1)*bucket_size, count))

# Show some samples of the parsed entities
def _show_samples(args, entities_file):
    if not os.path.exists(entities_file):
        logging.critical('File {} does not exist.'.format(entities_file))
        sys.exit(1)

    with args._open(entities_file, 'rb') as f:
        dataset = pickle.load(f)
    
    entities = dataset['entities']

    samples = random.sample(entities, min(args.show_samples, len(entities)))
    from kode_search.viewer import Viewer
    Viewer(samples).run()

# GPT api parameters
GPT_MODEL = 'gpt-3.5-turbo'
GPT_MAX_TOKENS = 3068 # leave 1024 tokens for the response. 4096 is the max token length for gpt-3.5-turbo

# call openai chat completion api with back-off
# returns a json string if successful, otherwise None
def _get_code_summary(args, code): 
    # If the code is too short, we don't need to summarize it
    if code.count('\n') < args.code_lines_threshold:
        logging.debug('Code is too short to summarize.')
        return ""
    
    # GPT chat completion api prompt.
    prompt_template = """What is the purpose of the following code and how does it achieve its objective? What are search keywords from the code?

    ```
    {}
    ```
    """
    
    prompt = prompt_template.format(code)
    messages = [{"role": "user", "content": prompt}]

    # check if the message is too long
    num_tokens = num_tokens_from_messages(messages)
    while num_tokens > GPT_MAX_TOKENS:
        logging.debug('Code is too long, removing some lines. {} tokens > {} tokens'.format(num_tokens, GPT_MAX_TOKENS))
        # estimate how many lines we should remove
        percent = GPT_MAX_TOKENS / num_tokens
        splitted_code_lines = code.split('\n')
        num_lines = int(len(splitted_code_lines) * percent)
        code = '\n'.join(splitted_code_lines[:num_lines])

        prompt = prompt_template.format(code)
        messages = [{"role": "user", "content": prompt}]
        num_tokens = num_tokens_from_messages(messages)

    # if the final message is too small, we don't need to summarize it
    if code.count('\n') < args.code_lines_threshold:
        logging.debug('Code is too short to summarize.')
        return None

    try_count = 0
    while try_count < args.openai_api_retries:
        try_count += 1
        try:
            response = openai.ChatCompletion.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message["content"]
        except Exception as e:
            logging.warning('GPT API Error: {}'.format(e))
            time.sleep(2**try_count)

    logging.error('Failed to get summary from GPT API after retrying.')
    return None

# Single thread version of summarizing entities
def _summarized_entity(args, entity):     
    if entity['content_type'] == 'code':
        summary = _get_code_summary(args, entity['content'])
        if summary is not None:
            entity['summary'] = summary

# Generate summaries for the parsed entities
# This step costs $$$.
def _generate_summaries(args, entities_file):
    logging.info('Generating summaries for entities in {}...'.format(entities_file))
    if not ask_user_confirmation("This step costs $$$, are you sure to continue?"):
        sys.exit(0)

    if not os.path.exists(entities_file):
        logging.critical('File {} does not exist.'.format(entities_file))
        sys.exit(1)

    with args._open(entities_file, 'rb') as f:
        dataset = pickle.load(f)

    entities = dataset['entities']

    if args.threads > 1:
        # Multi-thread version
        logging.info('Using {} threads to generate summaries...'.format(args.threads))
        with ThreadPoolExecutor(max_workers=args.threads) as t:
            _ = list(tqdm(t.map(partial(_summarized_entity, args), entities), total=len(entities)))
    else:
        for entity in tqdm(entities):
            _summarized_entity(args, entity)

    with args._open(entities_file, 'wb') as f:
        pickle.dump(dataset, f)

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
        with open(source_file, 'r') as f:
            source_code = f.read()
        parser = get_parser(SUPPORTED_FILE_EXTENSIONS[os.path.splitext(source_file)[1]])
        try:
            tree = parser.parse(bytes(source_code, 'utf8'))
        except UnicodeDecodeError:
            logging.error('UnicodeDecodeError when parsing {}'.format(source_file))
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

    # This code can be changed frequently.
    for e in tqdm(entities):
        if 'summary' in e and len(e['summary'].strip()) == 0:
            del e['summary']

    with args._open(entities_file, 'wb') as f:
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