import os
import json
from bs4 import BeautifulSoup
from collections import defaultdict
import loadbard
import ctoken
import sys
from psutil import virtual_memory
from collections import OrderedDict
import csearch
import time

# CONSTANTS
DATA_DIR = "DATA"
INDEX_DIR = "INDEX/"
INVERT_DIR = "INVERT/"
SEARCH_RESULTS = 5
Bard = loadbard.LoadBard()
jedi_archives = False


def write_bucket(inverted_index_bucket_ref, INDEX_DIR, docID):
    with open(f"{INDEX_DIR}{docID}.txt", "a") as f:
        # {token:{docID:{positions:}}}
        for token in sorted(inverted_index_bucket_ref.keys()):
            f.write(f"{token}~{dict(inverted_index_bucket_ref[token])}\n")


if not jedi_archives:
    # BEGIN
    print("<--------------COUNTING DOCUMENTS----------------->")
    # Count all documents in DATA
    abs_doc_count = 0
    abs_decrementer = 0
    for subdir, dirs, files in os.walk(DATA_DIR):
        for file in files:
            abs_doc_count += 1
    abs_decrementer = int(abs_doc_count)

    print("<--------------MAKING FRAGMENTED INDEX----------------->")
    # MEMORY
    document_store = list()
    inverted_index_bucket = defaultdict(lambda: defaultdict(list))  # {token:{docID:{positions:}}}
    buffer = virtual_memory().total/128
    tokenIDs = set()

    docID = 0
    if Bard:
        Bard.start("WALKING DATA")
    for subdir, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if ".json" in file:
                if sys.getsizeof(inverted_index_bucket) > buffer:
                    write_bucket(inverted_index_bucket, INDEX_DIR, docID)
                    inverted_index_bucket = defaultdict(lambda: defaultdict(list))
                with open(os.path.join(subdir, file), "r") as f:
                    jsonfile = json.load(f)
                soup = BeautifulSoup(jsonfile["content"], features="lxml")
                document_store.append(jsonfile["url"])
                position = 0
                for token in ctoken.tokenize(soup.text):
                    inverted_index_bucket[token][str(docID)].append(position)
                    position += 1
                    tokenIDs.add(token)
                percent_progress = float("{:.2f}".format(docID / len(files)))
                if Bard and percent_progress % 0.01 == 0:
                    Bard.update(f"{percent_progress * 100}% {subdir}/{file}", replace=True)
                docID += 1
    write_bucket(inverted_index_bucket, INDEX_DIR, docID)
    inverted_index_bucket.clear()
    sorted_tokenIDs = sorted(list(tokenIDs))

    with open(f"{INDEX_DIR}termID_map.json", "w") as f:
        json.dump(sorted_tokenIDs, f)

    print("<--------------MERGING FRAGMENTED INDEX----------------->")
    if Bard:
        Bard.end()
        #Bard.start("MERGING INDEX")
    index_file = 0
    file_objects = list()
    # open all fragment files for reading
    for subdir, dirs, files in os.walk(INDEX_DIR):
        for file in files:
            if ".txt" in file:
                file_objects.append(open(f"{INDEX_DIR}{file}", "r"))
    outfile = open(f"{INVERT_DIR}final.txt", "w")
    token_map_file = open(f"{INVERT_DIR}token_map.txt", "w")
    read_buffer = defaultdict(lambda: defaultdict(list))
    token_map = defaultdict(int) # {token: seek_index}
    tokenID = 0

    while tokenID < len(sorted_tokenIDs):
        for file in file_objects:
            thing = file.readline().split("~")
            if len(thing) != 2:
                print(thing)
                raise IndexError(thing)
            token = thing[0]
            dictionary = thing[1]
            docIDs_to_positions = json.loads(dictionary.replace("'", "\""))
            for docID, positions in docIDs_to_positions.items():
                for position in positions:
                    read_buffer[token][docID].append(position)
        min_tokenID = min(list(map(lambda x: sorted_tokenIDs.index(x), read_buffer.keys())))
        this_token = sorted_tokenIDs[min_tokenID]
        line = dict(read_buffer.pop(this_token))
        outfile_position = outfile.tell()
        outfile.write(f"{line}\n")
        token_map[this_token] = outfile_position
        print(f"{sorted_tokenIDs[min_tokenID]} FILEPOS:{outfile_position} {tokenID}/{len(sorted_tokenIDs)}")
        tokenID += 1

    json.dump(token_map, token_map_file)

    # close all files
    for file in file_objects:
        file.close()
    outfile.close()
    token_map_file.close()

def search(query_phrase):
    results = defaultdict(dict)
    with open(f"{INVERT_DIR}final.txt", 'r') as f:
        for toke in ctoken.tokenize(query_phrase):
            f.seek(token_map[toke]) # sends cursor to position
            docIDs_positions = f.readline() # reads until \n
            docIDs_positions = docIDs_positions.replace("'", "\"")
            results[toke] = json.loads(docIDs_positions)
    return results

with open(f"{INVERT_DIR}token_map.txt", "r") as f:
    token_map = json.load(f)

search_queries = ["cristina lopes", "machine learning", "ACM", "master of software engineering"]

for query in search_queries:
    time_search_start = time.perf_counter()
    results = search(query)
    common_docs = list()
    for term in results:
        keys = set(results[term].keys())
        common_docs.append(keys)
    intersect = set.intersection(*common_docs)
    if intersect:
        print(f"{query} intersect:", intersect)
    else:
        print(f"{query} no intersect:")
    # do querytime ranking
    time_search_end = time.perf_counter()
    print(f"Term '{query}' search took {time_search_end - time_search_start} seconds total.")

exit()

# deliverables top 5 results for each query and a photo of the cmd interface in action
