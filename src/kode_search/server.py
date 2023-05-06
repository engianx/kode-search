from flask import Flask, request, jsonify, render_template

import faiss
import html
import logging
import os
import pickle
import random
import sys

from annoy import AnnoyIndex

from kode_search.constants import FILE_EXTENSIONS, URL_TEMPLATES
from kode_search.index import validate_index
from kode_search.search import kode_search
from kode_search.utils import ask_user_confirmation

FAISS_INDEX = None
ANNOY_INDEX = None
INDEX_INFO = None
FILE_URL_TEMPLATE = None
INDEX_INFO_STR = None

DISTANCE_THRESHOLD = 1.0

app = Flask(__name__, template_folder='templates')

@app.route('/ask', methods=['POST'])
def ask():
    request_data = request.get_json()
    question = request_data['query']
    num_results = request_data['n']
    answers = _search(question, num_results)
    return jsonify(answers)

@app.route('/', methods=['GET'])
def home():
    global INDEX_INFO_STR
    return render_template('index.html', index_info=INDEX_INFO_STR, query="", results=None)

@app.route('/search', methods=['GET'])
def search():
    global FILE_URL_TEMPLATE, DISTANCE_THRESHOLD, INDEX_INFO

    num_results = request.args.get('n')
    if num_results is None:
        num_results = 10
    else:
        num_results = int(num_results)

    distance_threshold = request.args.get('d')
    if distance_threshold is not None:
        distance_threshold = float(distance_threshold)
    else:
        distance_threshold = DISTANCE_THRESHOLD

    dev_mode = request.args.get('dev') is not None
    query = request.args.get('query')
    if query is None or query.strip() == '':
        # Show some random samples.
        answers = _get_random_samples(num_results, dev_mode=dev_mode)
        query = "I'm Feeling Lucky" # placeholder
    else:
        # Perform actual search.
        query = query.strip()
        answers = _search(query, num_results, distance_threshold, dev_mode=dev_mode)

    if FILE_URL_TEMPLATE is not None:
        for answer in answers:
            answer['file_url'] = _generate_file_url(answer['file'], answer['line'])

    return render_template('index.html', index_info=INDEX_INFO_STR, query=query, results=answers)

def _generate_file_url(file, line):
    global FILE_URL_TEMPLATE
    
    # TODO: the file URL pattern should be configurable.
    # The file name may start with ./ or ../, which we need to remove.
    file = file.lstrip('./')
    return FILE_URL_TEMPLATE.format(file=file, line=line)

def _get_random_samples(num_results, dev_mode=False):
    global INDEX_INFO
    samples = [e for e in random.sample(INDEX_INFO['entities'], num_results)]
    answers = []
    for sample in samples:
        answer = {
            'file': html.escape(sample['file']),
            'line': sample['start_line'],
            'content_type': sample['content_type'],
            'content': sample['content'],
        }
        if dev_mode:
            answer['summary'] = html.escape(r['summary']) if 'summary' in r else '',
            answer['distance'] = 0.0,
        answers.append(answer)
    return answers

def _search(query, num_results, distance_threshold=DISTANCE_THRESHOLD, dev_mode=False):
    global FAISS_INDEX, ANNOY_INDEX, INDEX_INFO

    logging.info('Searching for: {}'.format(query))
          
    if FAISS_INDEX is not None:
        index = FAISS_INDEX
    elif ANNOY_INDEX is not None:
        index = ANNOY_INDEX
    else:
        raise Exception('Index not loaded.')

    # search() expects a list of queries.
    results, distances = kode_search(index, 
                                     INDEX_INFO, 
                                     query, 
                                     distance_threshold, 
                                     num_results, 
                                     return_distances=True)

    answers = []
    for i, r in enumerate(results):
        answer = {
            'file': html.escape(r['file']),
            'line': r['start_line'],
            'content_type': r['content_type'],
            'content': r['content'],
        }

        if dev_mode:
            answer['summary'] = html.escape(r['summary']) if 'summary' in r else '',
            answer['distance'] = distances[i],
        answers.append(answer)

    return answers

def _load_index(args, index_file, index_info_file):
    global FAISS_INDEX, ANNOY_INDEX, INDEX_INFO, FILE_URL_TEMPLATE, INDEX_INFO_STR

    validate_index(args, index_file, index_info_file)

    logging.info('Loading index...')

    with args._open(index_info_file, 'rb') as f:
        INDEX_INFO = pickle.load(f)

    if INDEX_INFO['model'] == 'openai' and not args.auto_confirm:
        if not ask_user_confirmation("This step costs $$$, are you sure to continue?"):
            sys.exit(0)

    index_type = INDEX_INFO['index_type']
    embedding_size = INDEX_INFO['embedding_size']

    if index_type == 'faiss':
        FAISS_INDEX = faiss.read_index(index_file)
    else: # index_type == 'annoy'
        ANNOY_INDEX = AnnoyIndex(embedding_size, 'angular')
        ANNOY_INDEX.load(index_file)

    INDEX_INFO_STR = '{}, {}, {}'.format(INDEX_INFO['model'], index_type, args.prefix)

    if args.url_tpl in URL_TEMPLATES:
        FILE_URL_TEMPLATE = URL_TEMPLATES[args.url_tpl]

    # warm up
    _search('warm up', 1)

    logging.info('Index loaded.')

def _start_server(args, index_file, index_info_file):
    _load_index(args, index_file, index_info_file)
    app.run(host='0.0.0.0', port=args.port, debug=args.log_level == 'DEBUG')

def run(args):
    index_file = os.path.join(args.repo_path, args.prefix + FILE_EXTENSIONS['index'])
    index_info_file = os.path.join(args.repo_path, args.prefix + '.index_info')

    _start_server(args, index_file, index_info_file)