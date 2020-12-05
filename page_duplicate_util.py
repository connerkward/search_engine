import json
from collections import defaultdict

def find_duplicates(docID_hash_dict):
    """
    Compares hashes in the hash dict
    :param docID_hash_dict:
    :return: set of duplicates as docID's
    """
    visited_hashes = set()
    docID_of_duplicates = set()
    for docID in docID_hash_dict.keys():
        if docID_hash_dict[docID] in visited_hashes:
            docID_of_duplicates.add(docID_hash_dict[docID])
        else:
            visited_hashes.add(docID_hash_dict[docID])
    return docID_of_duplicates


def gen_inverted_docID_map(docIDmap):
    ret = defaultdict(int)
    for index, docID in enumerate(docIDmap):
        ret[docID] = index
    return ret


if __name__ == '__main__':
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
    # INVERT
    token_seek_map_filename = f"{INVERT_DIR}token_seek.map"
    corpus_token_frequency_filename = f"{INVERT_DIR}corpus_token_freq.map"

    invert_docID_filename = f"{INDEX_DIR}inverted_docID.map"

    with open(docID_store_file_filename, "r") as f:
        inverted_docID_map = gen_inverted_docID_map(json.load(f))
        with open(invert_docID_filename, "w") as f:
            json.dump(inverted_docID_map, f)