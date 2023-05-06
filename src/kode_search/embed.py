import logging
import numpy
import os
import pickle
import random
import sys
from tqdm import tqdm

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from datetime import datetime
from pprint import pprint

from sentence_transformers import SentenceTransformer

from kode_search.constants import FILE_EXTENSIONS, EMBEDDING_MODELS
from kode_search.utils import ask_user_confirmation, openai_embed_content

def get_embeddings(model_name, inputs, show_progress_bar=False, batch_size=1, threads=1):
    """
    Get embeddings for a list of inputs. Returns a list of numpy array of embeddings.
    """
    if model_name == 'openai':
        if threads == 1: # special case when serving queries.
            embeddings = openai_embed_content(inputs)
            # the shape of embeddings is (#inputs, #embedding_size)
            return numpy.array(embeddings, dtype=numpy.float32)
        else:
            logging.info('Using {} threads to generate embeddings...'.format(threads))
            with ThreadPoolExecutor(max_workers=threads) as t:
                embeddings = list(tqdm(t.map(openai_embed_content, inputs, chunksize=batch_size), total=len(inputs)))  
            # the shape of embeddings is (#inputs, 1, #embedding_size)
            # we need to remove the middle dimension
            # and convert the list to numpy array
            # so the shape becomes (#inputs, #embedding_size)
            # and the type is numpy.float32
            return numpy.squeeze(numpy.array(embeddings, dtype=numpy.float32), axis=1)     
    else:
        # Load the model
        model = SentenceTransformer(EMBEDDING_MODELS[model_name])
        # Encode returns numpy already.
        return model.encode(inputs, 
                            convert_to_numpy=True,
                            show_progress_bar=show_progress_bar,
                            batch_size=batch_size)

def _show_samples(args, embeddings_file):
    if not os.path.exists(embeddings_file):
        logging.critical('Embeddings file {} does not exist.'.format(embeddings_file))
        sys.exit(1)

    with args._open(embeddings_file, 'rb') as f:
        dataset = pickle.load(f)

    entities = dataset['entities']
    embeddings = dataset['embeddings']

    num_samples = min(args.show_samples, len(entities))
    sampled_indices = [random.randint(0, len(entities)-1) for _ in range(num_samples)]

    for idx in sampled_indices:
        pprint(entities[idx], width=120)
        pprint(embeddings[idx], width=120)

def _show_info(args, embeddings_file):
    if not os.path.exists(embeddings_file):
        logging.critical('Embeddings file {} does not exist.'.format(embeddings_file))
        sys.exit(1)

    with args._open(embeddings_file, 'rb') as f:
        dataset = pickle.load(f)

    print('Embeddings info:')
    print('Project path:\t{}'.format(dataset['repo']))
    print('Model:\t{}'.format(dataset['model']))
    print('Type:\t{}'.format(dataset['type']))
    print('Creation date:\t{}'.format(dataset['creation_date']))
    print('Number of entities:\t{}'.format(len(dataset['entities'])))
    print('Embedding size:\t{}'.format(len(dataset['embeddings'][0])))

def _generate_embeddings(args, entities):
    # Prepare the input
    if args.embedding_type == 'code':
        embedding_input = [e['content'] for e in entities]
    elif args.embedding_type == 'summary':
        # Handle the case where the summary is empty
        embedding_input = [e['summary'] if 'summary' in e else e['content'] for e in entities]
    elif args.embedding_type == 'summary_and_code':
        embedding_input = [e['summary'] + '\n' + e['content'] if 'summary' in e else e['content'] for e in entities]
    else:
        logging.critical('Unsupported embedding type: {}'.format(args.embedding_type))
        sys.exit(1)

    if args.model_name == 'openai':
        if not ask_user_confirmation("This step costs $$$, are you sure to continue?"):
            sys.exit(0)

    # Generate the embeddings
    embeddings = get_embeddings(args.model_name,
                                embedding_input,
                                show_progress_bar=True,
                                batch_size=args.embedding_batch_size,
                                threads=args.threads)
    return embeddings

def _create_embeddings(args, entities_file, output_file):
    logging.info("Creating embeddings in {}, from {} ...".format(output_file, entities_file))
    # Load the entities from the .parse intermediate file
    with args._open(entities_file, 'rb') as f:
        entity_dataset = pickle.load(f)

    embeddings = _generate_embeddings(args, entity_dataset['entities'])

    dataset = {
        'repo': entity_dataset['repo'],
        'model': args.model_name,
        'type' : args.embedding_type,
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'entities': entity_dataset['entities'],
        'embeddings': embeddings,
    }

    with args._open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    logging.info('Embeddings created. saved in {}'.format(output_file))

# Take the .parse intermediate file and generate embeddings for each entity
# Save the embeddings in a file.
def embed(args):        
    embeddings_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['embed'])

    if args.show_samples:
        _show_samples(args, embeddings_file)
        return
    
    if args.info:
        _show_info(args, embeddings_file)
        return

    summary_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['summary'])
    if args.embedding_type in ['summary', 'summary_and_code']:
        input_file = summary_file
    else:
        # Generate embeddings from code using summary file as input if it exists, otherwise use the entity file
        input_file = summary_file if os.path.exists(summary_file) else os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['parse'])

    if args.run:
        _create_embeddings(args, input_file, embeddings_file)
        return
    
    print('No supported action specified, use one of the following options:')
    print('\t--info\tShow information about the embeddings file.')
    print('\t--show-samples\tShow some samples from the embeddings file.')
    print('\t--run\tCreate the embeddings file.')
