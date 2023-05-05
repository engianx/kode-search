import openai
import os
import sys
import time
import logging
import pickle
import shutil
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from kode_search.constants import FILE_EXTENSIONS
from kode_search.utils import ask_user_confirmation, num_tokens_from_messages
from kode_search.parse import show_info, show_samples

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

    # Call openai API with exponential back-off
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

# Recreate the entities file from the source code.
def _recreate(args, summary_file):
    logging.info('Recreating summary file {}...'.format(summary_file))
    if not os.path.exists(summary_file):
        logging.critical('File {} does not exist.'.format(summary_file))
        sys.exit(1)

    # Make a backup of the original file
    backup_file = summary_file + '.bak'
    shutil.copyfile(summary_file, backup_file)
    logging.info('Backup file saved to {}'.format(backup_file))

    with args._open(summary_file, 'rb') as f:
        dataset = pickle.load(f)

    entities = dataset['entities']

    # This code can be changed frequently.
    for e in tqdm(entities):
        if 'summary' in e and len(e['summary'].strip()) == 0:
            del e['summary'] 

    with args._open(summary_file, 'wb') as f:
        pickle.dump(dataset, f)

# Generate summaries for the parsed entities
# This step costs $$$.
def _generate_summaries(args, entities_file, output_file):
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

    with args._open(output_file, 'wb') as f:
        pickle.dump(dataset, f)  

def summarize(args):
    entities_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['parse'])
    summary_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['summary'])

    if args.info:
        show_info(args, summary_file)
        return
    
    if args.show_samples > 0:
        show_samples(args, summary_file)
        return

    if args.recreate:
        _recreate(args, summary_file)
        return

    if args.run:
        _generate_summaries(args, entities_file, summary_file)
        return
    
    print('No supported action specified, use one of the following options:')
    print('\t--info\tshow info about the parsed entities')
    print('\t--show-samples\tshow samples of the parsed entities')
    print('\t--recreate\trecreate the parsed entities')
    print('\t--run\tparse source code and save the parsed entities to a file')