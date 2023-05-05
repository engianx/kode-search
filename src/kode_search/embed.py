import logging
import os
import pickle
import random
import sys

from datetime import datetime
from pprint import pprint

from sentence_transformers import SentenceTransformer

from kode_search.constants import FILE_EXTENSIONS
from kode_search.utils import ask_user_confirmation

def get_embeddings(model_name, inputs, show_progress_bar=False, batch_size=1):
    # Load the model
    model = SentenceTransformer(model_name)    
    return model.encode(inputs, show_progress_bar=show_progress_bar, batch_size=batch_size)

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

    # Generate the embeddings
    embeddings = get_embeddings(args.model, embedding_input, show_progress_bar=True, batch_size=args.embedding_batch_size)

    return embeddings

def _create_embeddings(args, entities_file, output_file):
    logging.info("Creating embeddings in {}, from {} ...".format(output_file, entities_file))
    # Load the entities from the .parse intermediate file
    with args._open(entities_file, 'rb') as f:
        entity_dataset = pickle.load(f)

    embeddings = _generate_embeddings(args, entity_dataset['entities'])

    dataset = {
        'repo': entity_dataset['repo'],
        'model': args.model,
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
    entities_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['parse'])
    embeddings_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['embed'])

    if args.show_samples:
        _show_samples(args, embeddings_file)
        return
    
    if args.info:
        _show_info(args, embeddings_file)
        return
    
    if args.run:
        _create_embeddings(args, entities_file, embeddings_file)
        return
    
    print('No supported action specified, use one of the following options:')
    print('\t--info\tShow information about the embeddings file.')
    print('\t--show-samples\tShow some samples from the embeddings file.')
    print('\t--run\tCreate the embeddings file.')
