# File extension maps for the various intermidiate files
FILE_EXTENSIONS = {
    'parse' : '.entities',
    'summary' : '.summaries',
    'embed' : '.embeddings',
    'index' : '.index',
}

# Things to consider when choosing the model:
#   1. training dataset
#   2. model size
#   3. input token length
#   4. embedding size
EMBEDDING_MODELS = {
    'mpnet': 'all-mpnet-base-v2', # 128, 768
    'minilm': 'all-MiniLM-L6-v2', # 128, 384
    'roberta': 'all-distilroberta-v1', # 128, 768
    'openai': 'text-embedding-ada-002', # 8192, 1526
}

URL_TEMPLATES = {
    'android_art': 'https://cs.android.com/android/platform/superproject/+/master:art/{file};l={line}',
    'chromium_net': 'https://source.chromium.org/chromium/chromium/src/+/main:net/{file};l={line}',
}