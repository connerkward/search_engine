import os
# import json
from bs4 import BeautifulSoup
from collections import defaultdict
import loadbard
import ctoken
import sys
from psutil import virtual_memory
import time
import numpy as np
import orjson
import json
import math
import utils

# INSTANTIATION
Bard = loadbard.LoadBard()

# CONSTANTS
DATA_DIR = "/Users/connerward/Desktop/DEV"
# DATA_DIR = "DATA-S"
INDEX_DIR = "INDEX/"
INVERT_DIR = "INVERT/"
SEARCH_RESULTS = 5
# BUFFER = 500000  # 52428800 # 50 mb in bytes # currently 0.5 mb
INDEX_BUFFER = utils.get_size_directory(DATA_DIR) / 40  # 4000 == 0.5mb # 40 == 50mb
FINAL_MAX_LINES = 5000 # lines
print(f"BUFFER SIZE {INDEX_BUFFER / 1000000} mb")

# FILE NAMES
# INDEX
termID_map_name = f"{INDEX_DIR}termID.map"
docID_store_file_name = f"{INDEX_DIR}docid.map"
# INVERT
token_map_file_name = f"{INVERT_DIR}token_seek.map"


def write_bucket(inverted_index_bucket_ref, INDEX_DIR, docID):
    with open(f"{INDEX_DIR}{docID}.index", "a") as f:
        # {token:{docID:{positions:}}}
        for token in sorted(inverted_index_bucket_ref.keys()):
            f.write(f"{token}~{dict(inverted_index_bucket_ref[token])}\n")
        f.close()


def make_fragment_index():
    """
    Creates a fragmented index in the form of a few .txt files and a json for termID's and a json for docID's.
    :return:
    Creates docid.map, a json file that is a list of urls where the index is the docID
    Creates termID.map, a json file with a list of terms, where the termID is the index.
    Creates a few 'bucket' txt files, where the filename refers to the last added termID.
    Each bucket file contains alphabetical rows of "term~{document:[positions]}"
    These are partial datasets that must be merged in next step.
    """
    print("<--------------MAKING FRAGMENTED INDEX-----------------> 1/2")
    document_store = list()
    inverted_index_bucket = defaultdict(lambda: defaultdict(list))  # {token:{docID:{positions:}}}
    tokenIDs = set()
    docID = 0
    doc_count = utils.count_docs(DATA_DIR)

    # BARD IN!
    if Bard:
        Bard.start("WALKING DATA")
    for subdir, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if ".json" in file:
                # If bucket is at-size
                if sys.getsizeof(inverted_index_bucket) > INDEX_BUFFER:
                    print("\n<---------WRITING BUCKET--------->")
                    write_bucket(inverted_index_bucket, INDEX_DIR, docID)
                    inverted_index_bucket = defaultdict(lambda: defaultdict(list))
                # Open url-database file
                with open(os.path.join(subdir, file), "r") as f:
                    jsonfile = json.load(f)
                    f.close()
                document_store.append(jsonfile["url"])
                # TOKENIZE!
                soup = BeautifulSoup(jsonfile["content"], features="lxml")
                position = 0
                for token in ctoken.tokenize(soup.text):
                    inverted_index_bucket[token][str(docID)].append(position)
                    position += 1
                    tokenIDs.add(token)

                # BARD BARDING!
                percent_progress = float("{:.2f}".format(docID / doc_count))
                if Bard and percent_progress % 0.001 == 0:
                    Bard.update(f"{percent_progress * 100}% {sys.getsizeof(inverted_index_bucket)/1000000}mb {subdir}/{file}", replace=True)
                docID += 1
    print("<---------WRITING BUCKET--------->")
    write_bucket(inverted_index_bucket, INDEX_DIR, docID)
    inverted_index_bucket.clear()

    print("<---WRITING DOCUMENT STORE------->")
    with open(docID_store_file_name, "w") as f:
        json.dump(document_store, f)
        f.close()

    print("<-----WRITING TERM ID STORE------>")
    sorted_tokenIDs = sorted(list(tokenIDs))
    with open(termID_map_name, "w") as f:
        json.dump(sorted_tokenIDs, f)
        f.close()

    # BARD OUT!
    if Bard:
        Bard.end()


def merge_frag_index():
    """
    All .txt files of fragmented index are opened and a line is read from each. The 'minimum' term/token that
    exists in the read buffer (that which has the lowest index number in the tokenID map) is written.
    :return:
    """
    print("<--------------MERGING FRAGMENTED INDEX-----------------> 2/2")
    # load sorted tokenIDs
    with open(termID_map_name, "r") as f:
        sorted_tokenIDs = json.load(f)

    # BARD IN!
    if Bard:
        Bard.start("MERGING FRAGMENT DATA")

    # open all fragment files for reading
    file_objects = list()
    for subdir, dirs, files in os.walk(INDEX_DIR):
        for file in files:
            if ".txt" in file:
                file_objects.append(open(f"{INDEX_DIR}{file}", "r"))

    # IO Token Map and Final Index
    token_map_file = open(token_map_file_name, "w")
    # open outfile for final database and the map file
    outfile_ID = 0
    outfile_name = f"{INVERT_DIR}{outfile_ID}.findex"
    outfile = open(outfile_name, "w")
    outfile_lines = 0

    # establish read buffer, which contains terms that have been read but not been written to final index
    read_buffer = defaultdict(lambda: defaultdict(list))  # {token:{docID:[positions]}}
    # establish token map, which contains the seek index in the final output file
    token_map = defaultdict(lambda: defaultdict(int))  # {token:{file:file_seek_position}}
    tokenID = 0

    while tokenID < len(sorted_tokenIDs):
        # check if outfile is sufficient line size to merit new outfile
        if outfile_lines > FINAL_MAX_LINES:
            outfile.close()
            print("\n<---------NEW OUTFILE--------->")
            # start new outfile
            outfile_ID += 1
            outfile_name = f"{INVERT_DIR}{outfile_ID}.findex"
            outfile = open(outfile_name, "w")
        # iterate through all files and add next line to read buffer
        for file in file_objects:
            line = file.readline().split("~")
            if len(line) != 2:  # [term, positional_data]
                print(line)
                raise IndexError(line)
            token = line[0]
            dictionary = line[1]
            docIDs_to_positions = json.loads(dictionary.replace("'", "\"")) # required for embedded json
            for docID, positions in docIDs_to_positions.items():
                for position in positions:
                    read_buffer[token][docID].append(position)

        # find lowest token index of tokens currently in the read buffer
        min_tokenID = min(list(map(lambda x: sorted_tokenIDs.index(x), read_buffer.keys())))

        # get actual token from termID/tokenID
        this_token = sorted_tokenIDs[min_tokenID]

        # gets current cursor position in output file, then writes from read buffer.
        # Notes this outfile position in token map # {token:{file:file_seek_position}}
        outfile_position = outfile.tell()
        outfile.write(f"{dict(read_buffer.pop(this_token))}\n")
        token_map[this_token][outfile_name] = outfile_position
        # increment outfile line count
        outfile_lines += 1

        # BARD BARDING!
        percent_progress = float("{:.2f}".format(tokenID / len(sorted_tokenIDs)))
        if Bard and percent_progress % 0.01 == 0:
            Bard.update(f"{this_token}\t{percent_progress * 100}%\t{outfile_name}\tRead buffer:{len(read_buffer)}\t"
                        f"{sys.getsizeof(read_buffer)/1000000}mb",
                        replace=True)
        tokenID += 1

    # dump token map which contains {token:{file:file_seek_position}}
    json.dump(token_map, token_map_file)

    # close all files
    for file in file_objects:
        file.close()
    outfile.close()
    token_map_file.close()

    # BARD OUT!
    if Bard:
        Bard.end()


def normalization_term(term_scores):
    some_list = list()
    for term_score in term_scores:
        some_list.append(term_score * term_score)
    return math.sqrt(sum(some_list))


def search_multiple(search_queries_ref, token_map_ref, docid_store_ref):
    with open(final_outfile_name, "r") as f:
        ret_results = defaultdict(tuple)
        for query_phrase in search_queries_ref:
            time_search_start = time.perf_counter()
            term_docID_positions = defaultdict(dict) # {term{docID:[positions]}}
            tokenized_query = ctoken.tokenize(query_phrase, trim_stopwords=False)
            try:
                for toke in tokenized_query:
                    f.seek(token_map_ref[toke])  # sends cursor to position
                    term_docID_positions[toke] = orjson.loads(f.readline().replace("'", "\""))
            except KeyError:
                print("A token in query does not exist, skipping query.")
                continue

            # COMPUTE INTERSECTION
            intersect = set.intersection(*[set(term_docID_positions[term].keys()) # intersection docID's
                                           for term in term_docID_positions.keys()])

            # TF-IDF SCORING ---------------------------------
            query_normalized_weighted_tfidf_dict = defaultdict(lambda: defaultdict(float))  # {docID:{term:float}}
            document_normalized_weighted_tf_dict = defaultdict(lambda: defaultdict(float))  # {docID:{term:float}}
            final_scores = defaultdict(float)  # {docID:float}}
            for docID in intersect:
                # COMPUTE QUERY TF-IDF
                # weighted query tf == 1+log(term frequency in document)
                # for all terms compute tfidf for docID
                for term in tokenized_query:
                    # weighted tf = 1+log(term frequency in document)
                    weighted_tf = 1 + math.log(len(term_docID_positions[term][docID]))
                    # weighted idf = log(n document count / (how many documents the term appeared in)
                    weighted_idf = math.log(len(docID_store) / len(term_docID_positions[term]))
                    # weighted tf-idf score
                    tfidf = weighted_tf*weighted_idf
                    query_normalized_weighted_tfidf_dict[docID][term] = tfidf

                # NORMALIZE query-term scores
                n_term = normalization_term(query_normalized_weighted_tfidf_dict[docID].values())
                query_normalized_weighted_tfidf_dict[docID][term] = query_normalized_weighted_tfidf_dict[docID][term] / n_term

                # COMPUTE DOCUMENT TF
                # term frequency in document, list of score values() for all terms
                # weighted document tf == 1+log(term frequency in document)
                document_normalized_weighted_tf_dict[docID][term] = 1 + math.log(len(term_docID_positions[term][docID]))
                # NORMALIZE DOCUMENT TF
                n_term = normalization_term(document_normalized_weighted_tf_dict[docID].values())
                document_normalized_weighted_tf_dict[docID][term] = document_normalized_weighted_tf_dict[docID][term] / n_term

            # COMPUTE FINAL TERM-DOCUMENT RELEVANCE SCORE
            # sum(for all terms: query_term_score*document_normalized_weighted_tf_dict)
            for docID in intersect:
                some_list = list()
                for term in tokenized_query:
                    # TODO: make sure sum(for each term: term_query_tfidf*term_document_tfidf)is correct
                    some_list.append(query_normalized_weighted_tfidf_dict[docID][term] * document_normalized_weighted_tf_dict[docID][term])
                final_query_score = sum(some_list)
                final_scores[docid_store_ref[int(docID)]] = final_query_score

            # SORT AND SLICE RESULTS
            search_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[0:SEARCH_RESULTS]

            # print((time.perf_counter() - time_search_start)*1000)
            # for elem in search_results:
            #     print(elem)
            # exit()
            # score = defaultdict(lambda: defaultdict(float)) # {docID:{score_params:float}}
            # final_score = defaultdict(float)
            # for docID in intersect:
            #     for token in tokenized_query:
            #         # TODO: remove matches, is a ridiculous number
            #         score[docID]["matches"] += len(term_docID_positions[token][docID])/10000
            #         # TODO: check if any query words are bolded in document, will be between 0 and n
            #         # score[docID]["bolds_count"] += term_docID_positions[token][docID]["bolds_count"]
            #         score[docID]["bolds_count"] += 1
            #         # TODO: add TF-IDF score for each word (will be below 1)
            #         # TODO: does tf-idf have to be done at search time for the whole query or just each word?
            #         # score[docID]["tfidf"] += term_docID_positions[token][docID]["tfidf"] # index per word tfidf
            #         # score[docID]["tfidf"] = compute_tfidf() # newly computed tf-idf
            #         score[docID]["tfidf"] = 0.15  # newly computed tf-idf
            #     # TODO: check if query exists in adjacent positional order in document, score for each ngram
            #     # score[docID]["ngrams"] \
            #     #     += position_intersect_count([term_docID_positions[token][docID] for token in tokenized_query])
            #
            #     # TODO: recompute score for docID
            #     final_score[docid_store_ref[int(docID)]] = sum(score[docID].values())
            # # sorted and slice top results
            # search_results = sorted(final_score.items(), key=lambda x: x[1], reverse=True)[0:SEARCH_RESULTS]

            time_taken = time.perf_counter() - time_search_start
            if intersect:
                ret_results[query_phrase] = (search_results, time_taken)
            else:
                ret_results[query_phrase] = ([], time_taken)
    return ret_results


if __name__ == '__main__':
    frag_index_exists = False
    all_index_exists = False

    # Build Indexes
    if not frag_index_exists:
        make_fragment_index()
        frag_index_exists = True

    if not all_index_exists:
        merge_frag_index()
        all_index_exists = True

    exit()
    # add maps back to memory
    with open(token_map_file_name, "r") as f:
        from_file_token_map = json.load(f)
    with open(docID_store_file_name, "r") as f:
        docID_store = json.load(f)

    # Search
    search_queries = ["machine learning", "cristina lopes", "machine learning", "ACM", "master of software engineering"]
    # search_queries = ["master of software engineering"]
    results = search_multiple(search_queries, from_file_token_map, docID_store)
    for query in results.keys():
        result = results[query][0]
        time_taken = results[query][1]
        print(f"Query '{query}' took {time_taken*1000} milliseconds.")
        for index, url in enumerate(result):
            print(f"{index + 1}. {url}")
        print()