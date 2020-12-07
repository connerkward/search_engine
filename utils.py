import os
import json
from collections import defaultdict
import cpagerank

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
inverted_bolds_filename = f"{INVERT_DIR}inverted_bolds.map"
supplemental_info_filename = f"{INDEX_DIR}bold-links.map"
inverted_bolds_filename = f"{INVERT_DIR}inverted_bolds.map"
pagerank_filename = f"{INVERT_DIR}pagerank.map"


def get_size_directory(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def count_docs(DATA_DIR):
    abs_doc_count = 0
    for subdir, dirs, files in os.walk(DATA_DIR):
        for file in files:
            abs_doc_count += 1
    return abs_doc_count


def make_docID_lib(docID_store_file_name, DATA_DIR):
    docIDs = list()
    with open(docID_store_file_name, "a") as f1:
        for subdir, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if ".json" in file:
                    with open(f"{subdir}/{file}", "r",) as f:
                        url = json.load(f)["url"]
                        docIDs.append(url)
                        f1.write(url+"\n")
    return docIDs


def make_invert_bolds_term_docID(bolds_filename, output_filename):
    bolds = defaultdict(list) # {term:[docID]}
    with open(bolds_filename, "r") as f:
        supps = json.load(f) # {docID:{"bolds":[word]}}
        for docID in supps:
            for term in supps[docID]["bolds"]:
                bolds[term].append(docID)
    with open(f"{output_filename}", "w") as f:
        json.dump(bolds, f)


def make_pagerank_lib(supps_filename, docID_fielname, output_filename):
    with open(docID_fielname, "r") as f:
        docID_store = json.load(f)  # [docID]
    with open(invert_docID_filename, "r") as f:
        invert_docID_map = json.load(f)  # {docID: ??}
    with open(supps_filename, "r") as f:
        bolds_links_store = json.load(f)  # {docID:{"bolds":[word]}}
        docID_links = {docID_store[int(docID)]: bolds_links_store[docID]["links"]
                       for docID in bolds_links_store.keys() if bolds_links_store[docID]["links"]}
        sorted_pagerank = sorted(cpagerank.pagerank(docID_links, invert_docID_map).items(), key=lambda x: x[1],
                                 reverse=True)
    with open(output_filename, "w") as f:
        json.dump(sorted_pagerank, f)


if __name__ == '__main__':
    make_invert_bolds_term_docID(supplemental_info_filename, inverted_bolds_filename)
    make_pagerank_lib(supplemental_info_filename, docID_store_file_filename, pagerank_filename)