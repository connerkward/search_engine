import json

INDEX_DIR = "INDEX\\"
docID_hash_filename = f"{INDEX_DIR}docID_hash.map"


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


if __name__ == '__main__':
    with open(docID_hash_filename, "r") as f:
        docID_hash = json.load(f)
        print("documents: ", len(docID_hash))
        duplicates = find_duplicates(docID_hash)
        print("duplicates: ", len(duplicates))