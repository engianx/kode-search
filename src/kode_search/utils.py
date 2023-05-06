import logging
import openai
import tiktoken
import time

from functools import partial

def ask_user_confirmation(msg):
    while True:
        response = input('{} [y/n] '.format(msg)).lower()
        if response == 'y':
            print("Continuing ...")
            return True
        elif response == 'n':
            print("Aborting ...")
            return False
        else:
            print("Invalid response: {}".format(response))

def _num_tokens_from_chat_messages(messages, model):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning("Warning: model not found. Using cl100k_base encoding.")
    
    encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        logging.warning("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return _num_tokens_from_chat_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        logging.warning("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return _num_tokens_from_chat_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
    

# Number of times to retry the API call
API_CALL_RETRIES = 3
def _call_api_with_backoff(api_call, *args, **kwargs):
    """Calls an API with exponential backoff."""
    try_count = 0
    while try_count < API_CALL_RETRIES:
        try_count += 1
        try:
            return api_call(*args, **kwargs)
        except Exception as e:
            logging.warning('GPT API Error: {}'.format(e))
            time.sleep(2**try_count)

    return None

# GPT api parameters
GPT_MODEL = 'gpt-3.5-turbo'
GPT_MAX_TOKENS = 3068 # leave 1024 tokens for the response. 4096 is the max token length for gpt-3.5-turbo
# At least 10 lines of code are needed to summarize
MINIMUM_CODE_LINES = 10
def openai_summarize_code(code): 
    """Call openai chat completion api with back-off, returns a json string if successful, otherwise None"""
    # If the code is too short, we don't need to summarize it
    if code.count('\n') < MINIMUM_CODE_LINES:
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
    num_tokens = _num_tokens_from_chat_messages(messages, model=GPT_MODEL)
    while num_tokens > GPT_MAX_TOKENS:
        logging.debug('Code is too long, removing some lines. {} tokens > {} tokens'.format(num_tokens, GPT_MAX_TOKENS))
        # estimate how many lines we should remove
        percent = GPT_MAX_TOKENS / num_tokens
        splitted_code_lines = code.split('\n')
        num_lines = int(len(splitted_code_lines) * percent)
        code = '\n'.join(splitted_code_lines[:num_lines])

        prompt = prompt_template.format(code)
        messages = [{"role": "user", "content": prompt}]
        num_tokens = _num_tokens_from_chat_messages(messages, model=GPT_MODEL)

    # if the final message is too small, we don't need to summarize it
    if code.count('\n') < MINIMUM_CODE_LINES:
        logging.debug('Code is too short to summarize.')
        return None

    response = _call_api_with_backoff(partial(openai.ChatCompletion.create, model=GPT_MODEL, messages=messages, temperature=0))
    if response is None:
       logging.error('Failed to summarize code via GPT API.')

    return response.choices[0].message["content"] if response else None


GPT_EMBEDDING_MODEL = 'text-embedding-ada-002'
GPT_EMBEDDING_TOKEN_LIMIT = 8191
def openai_embed_content(input):
    """
    Call openai embedding api with back-off, returns embeddings if successful, otherwise None
    input: a string or a list of strings, each has to be less than 8192 tokens.
    return: a list of embeddings, or a list or None. The length of the returned list is the same as the input.
    """
    # reduce the input size if it's too long
    if isinstance(input, str):
        input = [input]

    encoder = tiktoken.get_encoding("cl100k_base")

    for i in range(len(input)):
        text = input[i]
        # check if the input exceeds the token limit
        num_tokens = len(encoder.encode(text))
        while num_tokens > GPT_EMBEDDING_TOKEN_LIMIT:
            logging.debug('Code is too long, removing some lines. {} tokens > {} tokens'.format(num_tokens, GPT_EMBEDDING_TOKEN_LIMIT))
            # estimate how many lines we should remove
            percent = GPT_EMBEDDING_TOKEN_LIMIT / num_tokens
            end_pos = int(len(text) * percent)
            text = text[:end_pos]
            num_tokens = len(encoder.encode(text))
        input[i] = text

    response = _call_api_with_backoff(partial(openai.Embedding.create, input=input, model=GPT_EMBEDDING_MODEL))
    if response is None:
        logging.error('Failed to create embedding via GPT API.')

    #response = {
    #    "data": [
    #        {
    #            'embedding' : [1] * (GPT_EMBEDDING_TOKEN_LIMIT + 1),
    #            'index' : 0,
    #        }] * len(input),
    #   "model": "text-embedding-ada-002", 
    #}
    # ada-002 response format:
    # {
    # "data": [
    #    {
    #    "embedding": [
    #        ...
    #        -0.0114
    #    ],
    #    "index": 0,
    #    "object": "embedding"
    #    }
    # ],
    # "model": "text-embedding-ada-002",
    # "object": "list"
    # }
    return [d['embedding'] for d in response['data']] if response else [[0] * (GPT_EMBEDDING_TOKEN_LIMIT + 1) ] * len(input)