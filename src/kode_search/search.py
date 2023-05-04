import faiss
import logging
import os
import pickle
import sys

import numpy as np

from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from kode_search.index import validate_index

from kode_search.constants import FILE_EXTENSIONS

# It takes an array of queries and returns an array of embeddings.
def _get_query_embedding(model_name, queries):
    model = SentenceTransformer(model_name)
    return model.encode(queries, convert_to_tensor=True)

# Expose the function to be used by the server.
# It takes a query, returns a list of entities.
def kode_search(index, index_info, query, distance_threshold, num_results, return_distances=False):
    logging.info('Searching "{}" using {} index, distrance_threshold={}.'.format(query, index_info['index_type'], distance_threshold))
    query_embeddings = _get_query_embedding(index_info['model'], [query])        

    if index_info['index_type'] == 'faiss':
        # faiss search accepts an array of vectors at a time.
        D, I = index.search(np.array(query_embeddings), num_results)
        D = D[0]
        I = I[0]
    elif index_info['index_type'] == 'annoy':
        # annoy get_nns_by_vector accepts only one vector at a time.
        I, D = index.get_nns_by_vector(np.array(query_embeddings[0]), num_results, include_distances=True)
    else:
        raise Exception('Unknown index type: {}'.format(index_info['index_type']))

    logging.debug('D: {}'.format(D))
    logging.debug('I: {}'.format(I))

    result_indices = [I[i] for i in range(len(I)) if D[i] <= distance_threshold]
    if return_distances:
        return ([index_info['entities'][i] for i in result_indices], [D[i] for i in result_indices])
    else:
        return [index_info['entities'][i] for i in result_indices]

def _do_search(args, index_file, index_info_file):
    validate_index(args, index_file, index_info_file)

    with args._open(index_info_file, 'rb') as f:
        index_info = pickle.load(f)

    index_type = index_info['index_type']
    embedding_size = index_info['embedding_size']
    num_results = max(args.show_samples, 3)

    if index_type == 'faiss':
        index = faiss.read_index(index_file)
    elif index_type == 'annoy':
        index = AnnoyIndex(embedding_size, 'angular')
        index.load(index_file)
    else:
        raise Exception('Unknown index type: {}'.format(index_type))
    
    query = ' '.join(args.query)
    results = kode_search(index, 
                          index_info, 
                          query,
                          args.distance_threshold,
                          num_results,
                          include_distances=False)

    if len(results) == 0:
        print('No results found.')
    else:
        print('Found {} answers.'.format(len(results)))
        from kode_search.viewer import Viewer
        Viewer(results).run()

def search(args):
    index_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['index'])
    index_info_file = os.path.join(args.repo_path, args.prefix + '.index_info')

    if not args.query:
        print('Query is empty.')
        sys.exit(1)

    _do_search(args, index_file, index_info_file)