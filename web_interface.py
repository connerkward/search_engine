from flask import Flask, request
import main
import os
import ctoken
import json
import utils
import page_duplicate_util
from nltk.corpus import stopwords
import cpagerank
app = Flask(__name__)

# FILE NAMES
DATA_DIR = "C:\\Users\\Conner\\Desktop\\developer\\DEV"
# DATA_DIR = ""
INDEX_DIR = "INDEX\\"
INVERT_DIR = "INVERT\\"
# INDEX
termID_map_filename = f"{INDEX_DIR}termID.map"
docID_store_file_filename = f"{INDEX_DIR}docid.map"
supplemental_info_filename = f"{INDEX_DIR}bold-links.map"
docID_hash_filename = f"{INDEX_DIR}docID_hash.map"
invert_docID_filename = f"{INDEX_DIR}inverted_docID.map"
# INVERT
token_seek_map_filename = f"{INVERT_DIR}token_seek.map"
corpus_token_frequency_filename = f"{INVERT_DIR}corpus_token_freq.map"

# CONSTANTS
SEARCH_RESULTS = 5
FILE_COUNT = utils.count_docs(DATA_DIR)
INDEX_BUFFER = (utils.get_size_directory(DATA_DIR) / FILE_COUNT) * 55  # size/1000 BEST FOR FULL DATA SET
FINDEX_MAX_LINES = 50000  # lines
FINDEX_BUFFER = 10000

print("Loading maps of meaning...")
# add maps back to memory
with open(token_seek_map_filename, "r") as f:
    from_file_token_map = json.load(f)
with open(docID_store_file_filename, "r") as f:
    docID_store = json.load(f)
with open(docID_hash_filename, "r") as f:
    docID_hash = json.load(f)
with open(supplemental_info_filename, "r") as f:
    bolds_links_store = json.load(f)
with open(corpus_token_frequency_filename, "r") as f:
    corpus_token_frequency = json.load(f)
with open(invert_docID_filename, "r") as f:
    invert_docID_map = json.load(f)

stopwords = [ctoken.tokenize(i)[0] for i in stopwords.words('english')]
sorted_corpus = [(term, count)
                 for term, count
                 in sorted(corpus_token_frequency.items(), key=lambda x: x[1], reverse=True)
                 if term not in stopwords]
duplicate_docIDs = page_duplicate_util.find_duplicates(docID_hash)

docID_links = {docID_store[int(docID)]: bolds_links_store[docID]["links"]
               for docID in bolds_links_store.keys() if bolds_links_store[docID]["links"]}
sorted_pagerank = sorted(cpagerank.pagerank(docID_links, invert_docID_map).items(), key=lambda x: x[1], reverse=True)

PRINT_TERMS = 10

# open all files in FINDEX
findex_file_objects = dict()
for subdir, dirs, files in os.walk(INVERT_DIR):
    for file in files:
        if ".findex" in file:
            f = open(f"{INVERT_DIR}{file}", "r")
            findex_file_objects[f.name] = f
template = '<form method="POST">\n\t<h1>SEARCH</h1>\n\t<input name="search">\n\t<input type="submit">\n</form>'

@app.route('/')
def my_form():
    return template

@app.route('/', methods=['POST'])
def my_form_post():
    query = [request.form['search']]
    results = main.search_multiple(query, from_file_token_map, docID_store,
                                   findex_file_objects, duplicate_docIDs, bolds_links_store, sorted_pagerank)
    result_str = list()
    for query in results.keys():
        result = results[query][0]
        time_taken = results[query][1]
        result_str.append(f"Query '{query}' took {time_taken} seconds.<br>")
        for index, url in enumerate(result):
            result_str.append(f"{index + 1}. {url[0]} score:{url[1]}<br>")
        result_str.append("<br>")
    result_str = "".join(result_str)
    return template + result_str

if __name__ == '__main__':
    app.run()