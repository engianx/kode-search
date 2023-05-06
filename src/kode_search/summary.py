import os
import sys
import logging
import pickle
import shutil
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

from kode_search.constants import FILE_EXTENSIONS
from kode_search.utils import ask_user_confirmation, openai_summarize_code
from kode_search.parse import show_info, show_samples

# Single thread version of summarizing entities
def _summarized_entity(entity):     
    if entity['content_type'] == 'code':
        summary = openai_summarize_code(entity['content'])
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
            _ = list(tqdm(t.map(_summarized_entity, entities), total=len(entities)))
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