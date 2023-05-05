import faiss
import logging
import os
import sys
import pickle

import numpy as np

from annoy import AnnoyIndex
from datetime import datetime
from kode_search.constants import FILE_EXTENSIONS

def validate_index(args, index_file, index_info_file):
    if not os.path.exists(index_file):
        logging.critical('Index file {} does not exist.'.format(index_file))
        sys.exit(1)

    if not os.path.exists(index_info_file):
        logging.critical('Index info file {} does not exist.'.format(index_info_file))
        sys.exit(1)

    with args._open(index_info_file, 'rb') as f:
        index_info = pickle.load(f)

    index_type = index_info['index_type']
    embedding_size = index_info['embedding_size']
    num_entities = index_info['num_entities']

    assert num_entities == len(index_info['entities'])
    assert index_info['creation_date'] is not None
    assert index_type in ['faiss', 'annoy']

    if index_type == 'faiss':
        index = faiss.read_index(index_file)
        assert index.ntotal == num_entities
        assert index.d == embedding_size
    else: # index_type == 'annoy'
        index = AnnoyIndex(embedding_size, 'angular')
        index.load(index_file)
        assert index.get_n_items() == num_entities

def _show_info(args, index_file, index_info_file):
    validate_index(args, index_file, index_info_file)
    
    with args._open(index_info_file, 'rb') as f:
        index_info = pickle.load(f)

    index_type = index_info['index_type']
    embedding_size = index_info['embedding_size']

    # print index file size
    print('Index info:')
    print('\tcreation date:\t{}'.format(index_info['creation_date']))
    print('\tmodel:\t{}'.format(index_info['model']))
    print('\tindex type:\t{}'.format(index_type))
    print('\tfile size:\t{} bytes'.format(os.path.getsize(index_file)))

    if index_type == 'faiss':
        index = faiss.read_index(index_file)
        print('\tnumber of entities:\t{}'.format(index.ntotal))
        print('\tembedding size:\t{}'.format(index.d))
    else: # index_type == 'annoy'
        index = AnnoyIndex(embedding_size, 'angular')
        index.load(index_file)
        print('\tnumber of entities:\t{}'.format(index.get_n_items()))
        print('\tnmbedding size:\t{}'.format(embedding_size))

def _create_index(args, embeddings_file, index_file, index_info_file):
    logging.info("Creating index in {}, from {} ...".format(index_file, embeddings_file))

    if args.index_type not in ['faiss', 'annoy']:
        logging.critical('Unsupported index type: {}'.format(args.index_type))
        sys.exit(1)    

    if not os.path.exists(embeddings_file):
        logging.critical('Embeddings file {} does not exist.'.format(embeddings_file))
        sys.exit(1)

    with args._open(embeddings_file, 'rb') as f:
        dataset = pickle.load(f)
    
    embeddings = dataset['embeddings']
    index_info = {
        'model' : dataset['model'],
        'index_type' : args.index_type,
        'creation_date' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_entities': len(embeddings),
        'embedding_size': len(embeddings[0]),
        'embeddings' : embeddings,
        'entities': dataset['entities'],
    }

    if args.index_type == 'faiss':
        logging.info('Creating faiss index ...')
        faiss_index = faiss.IndexFlatL2(len(embeddings[0]))
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, index_file)
    else: # args.index_type == 'annoy'
        logging.info('Creating annoy index ...')
        annoy_index = AnnoyIndex(len(embeddings[0]), 'angular')
        for i, embedding in enumerate(embeddings):
            annoy_index.add_item(i, embedding)
        annoy_index.build(args.annoy_num_trees)
        annoy_index.save(index_file)

    with args._open(index_info_file, 'wb') as f:
        pickle.dump(index_info, f)

    logging.info('Done. Index info saved in {}'.format(index_info_file))

def index(args):
    embeddings_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['embed'])
    index_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['index'])
    index_info_file = os.path.join(args.repo_path, args.prefix + '.index_info')
    
    if args.info:
        _show_info(args, index_file, index_info_file)
        return

    if args.run:
        _create_index(args, embeddings_file, index_file, index_info_file)
        return
    
    print('No supported action specified. Use one of the following options:')
    print('\t--info\t\tShow index info')
    print('\t--run\t\tCreate index')