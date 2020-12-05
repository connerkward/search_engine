from collections import defaultdict
import json


def pagerank(page_outlinks, invert_docID_map):
    inwards = defaultdict(list)
    for url in page_outlinks.keys():
        for link in page_outlinks[url]:
            if link != url:
                inwards[link].append(url)

    base_rank = 1/len(page_outlinks)
    page_ranks = defaultdict(lambda: base_rank)
    page_ranks2 = defaultdict(int)
    page_ranks3 = defaultdict(int)
    for url in inwards: # A
        for inwards_url in inwards[url]: # C
            page_ranks2[url] += page_ranks[inwards_url] / len(page_outlinks[inwards_url])
    for url in inwards: # A
        for inwards_url in inwards[url]: # C
            try:
                page_ranks3[invert_docID_map[url]] += page_ranks2[inwards_url] / len(page_outlinks[inwards_url])
            except ValueError and KeyError:
                pass

    if sum(page_ranks3.values()) != 1:
        pass
        # print("sum:",sum(page_ranks2.values()))

    return page_ranks3 #{url:pagerank}

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
    invert_docID_filename = f"{INDEX_DIR}inverted_docID.map"
    # INVERT
    token_seek_map_filename = f"{INVERT_DIR}token_seek.map"
    corpus_token_frequency_filename = f"{INVERT_DIR}corpus_token_freq.map"
    with open(invert_docID_filename, "r") as f:
        invert_docID = json.load(f)

    with open(docID_store_file_filename, "r") as f:
        docID_store = json.load(f)

    with open(supplemental_info_filename, "r") as f:
        supps = json.load(f)
        docID_links = {docID_store[int(docID)]:supps[docID]["links"] for docID in supps.keys() if supps[docID]["links"]}
        print(sorted(pagerank(docID_links, invert_docID).items(), key=lambda x: x[1], reverse=True))

