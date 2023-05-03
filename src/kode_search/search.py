import faiss
import os
import pickle
import sys

import numpy as np

from annoy import AnnoyIndex
from pprint import pprint
from sentence_transformers import SentenceTransformer
from kode_search.index import validate_index

from kode_search.constants import FILE_EXTENSIONS

def _get_query_embedding(model_name, query):
    model = SentenceTransformer(model_name)
    return model.encode(query, convert_to_tensor=True)

def _faiss_search(args, index_file, query_embedding, num_results):
    index = faiss.read_index(index_file)
    D, I = index.search(np.array(query_embedding), num_results)
    if args.verbose > 0:
        print('D: {}'.format(D))
        print('I: {}'.format(I))
    result_indices = [I[0][i] for i in range(len(I[0])) if D[0][i] <= args.distance_threshold]
    return result_indices

def _annoy_search(args, index_file, query_embedding, num_results):
    index = AnnoyIndex(len(query_embedding), 'angular')
    index.load(index_file)
    I = index.get_nns_by_vector(np.array(query_embedding), num_results, include_distances=False)
    if args.verbose > 0:
        print('I: {}'.format(I))
    return I

def _search(args, index_file, index_info_file):
    validate_index(index_file, index_info_file)

    with args._open(index_info_file, 'rb') as f:
        index_info = pickle.load(f)

    model_name = index_info['model']
    index_type = index_info['index_type']
    embedding_size = index_info['embedding_size']

    query_embedding = _get_query_embedding(model_name, args.query)
    num_results = max(args.show_samples, 3)

    if index_type == 'faiss':
        result_indices = _faiss_search(args, index_file, query_embedding, num_results)
    else:
        result_indices = _annoy_search(args, index_file, query_embedding, num_results)

    if len(result_indices) == 0:
        print('No results found.')
        return

    print('Found {} results.'.format(len(result_indices)))
    entities = index_info['entities']
    for i in result_indices:
        pprint(entities[i], width=120)

def search(args):
    index_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['index'])
    index_info_file = os.path.join(args.repo_path, args.prefix + '.index_info')

    if not args.query:
        print('Query is empty.')
        sys.exit(1)

    _search(args, index_file, index_info_file)