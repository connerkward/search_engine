import os
import ctoken
from nltk.stem import PorterStemmer
import time
import csearch
import index
import json
# TOOLING
stemmer = PorterStemmer()  # stemmer.stem()
tokenizer = ctoken.tokenize

# CONSTANTS
DATA_DIR = "/Users/connerward/Desktop/DEV"

INDEX_DIR = "INDEX/"
INVERT_DIR = "INVERT/"

INDEX_CHUNK_SIZE = 200
INVERT_BUCKET_SIZE = 1000
SEARCH_RESULTS = 5

TFIDF = False
REPORT = False

INDEX_TO_FILE = True
INVERT_TO_FILE = True

INDEX_EXISTS = False
INVERT_EXISTS = False
DOCSTORE_EXISTS = False

# INSTANTIATION
document_store = list()  # list of document urls. index number is doc id
no_token_debug_store = list()
wordcount_index = dict()  # {docID:{ { bolds:[bold words] word_dict:{} } }}
inverted_index = dict()  # {word: {docID: {word count in document, bold_bool}}
abs_doc_count = 0
abs_decrementer = 0

# BEGIN
print("<--------------PROCESSING----------------->")
time1 = time.perf_counter()

# Count all documents in DATA
for subdir, dirs, files in os.walk(DATA_DIR):
    for file in files:
        abs_doc_count += 1
abs_decrementer = int(abs_doc_count)

# INDEX
if not INDEX_EXISTS:
    if INDEX_TO_FILE:  # index does not exist, and index intends to go to file
        index.make_index(wordcount_index_ref=wordcount_index, document_store=document_store,
                         abs_decrementer=abs_decrementer, INDEX_TO_FILE=INDEX_TO_FILE,
                         tokenizer=tokenizer, DATA_DIR=DATA_DIR, INVERT_EXPORT_DIR=INDEX_DIR,
                         INDEX_CHUNK_SIZE=INDEX_CHUNK_SIZE, abs_docs=abs_doc_count, )
        INDEX_EXISTS = True

# INVERT INDEX
if not INVERT_EXISTS and INDEX_EXISTS:
    index.invert_from_to_file_simple(wordcount_index_ref=wordcount_index,
                                     inverted_index_ref=inverted_index,
                                     INDEX_DIR=INDEX_DIR,
                                     INVERT_EXPORT_DIR=INVERT_DIR,
                                     INVERT_BUCKET_SIZE=INVERT_BUCKET_SIZE,
                                     INVERT_TO_FILE=INVERT_TO_FILE,
                                     tfidf=TFIDF
                                     )

print("<------------LOADING INVERT----------------->")
# load invert index
inverted_index = index.load_invert(INVERT_DIR=INVERT_DIR, inverted_index_ref=inverted_index)
document_store = index.load_doc_store()
# word_store = index.load_word_store()

if REPORT:
    # Validating absolute document count -> inverted_index document count
    # inverted index {word: {docID: {word positions, bold_bool}}
    time2 = time.perf_counter()
    inverted_doc_count = set()
    for val in inverted_index.items():
        for key in val[1].keys():
            inverted_doc_count.add(key)
    print("<--------------REPORT----------------->")
    print(f"All processes took {time2 - time1} seconds.")
    print(f"absolute original document count: {abs_doc_count}")
    print(f"inverted index document count: {len(inverted_doc_count)}")  # 1308 docs 1297 in index 1304 accounted
    print(f'Document difference (empty documents): {abs_doc_count - len(inverted_doc_count)} documents.')

    unique_tokens = len(inverted_index)
    print(f"{unique_tokens} unique tokens.")

    size1 = sum(os.path.getsize(os.getcwd() + "/" + INDEX_DIR + f) for f in os.listdir(INDEX_DIR)) / 1000
    size2 = sum(os.path.getsize(os.getcwd() + "/" + INVERT_DIR + f) for f in os.listdir(INVERT_DIR)) / 1000
    print(f"Size of INDEX: {size1}kB")
    print(f"Size of INVERTED INDEX: {size2}kB")

print("<--------------SEARCH----------------->")
searches = ["Sudeep Pasricha", "cristina lopes", "machine learning", "ACM", "master of software engineering",]
for search in searches:
    time_search_start = time.perf_counter()
    rank = 1
    for docID in [x[0] for x in csearch.search(search, invert_ref=inverted_index)[0:SEARCH_RESULTS]]:
        print(f"RANK: {rank} DOCID: {docID} URL:{document_store[int(docID)]}")
        rank += 1
    time_search_printed = time.perf_counter()
    print(f"Term '{search}' search took {time_search_printed - time_search_start} seconds total.")

