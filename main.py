import os
from bs4 import BeautifulSoup, SoupStrainer
from collections import defaultdict
import loadbard
import ctoken
import sys
from psutil import virtual_memory
import time
import orjson
import json
import math
import utils
import warnings
import page_duplicate_util
import hashlib
from nltk.corpus import stopwords
import validators
import csimhash
import cpagerank

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# INSTANTIATION
Bard = loadbard.LoadBard()

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
inverted_bolds_filename = f"{INVERT_DIR}inverted_bolds.map"
pagerank_filename = f"{INVERT_DIR}pagerank.map"

# CONSTANTS
FILE_COUNT = utils.count_docs(DATA_DIR)
# INDEX_BUFFER = (utils.get_size_directory(DATA_DIR) / FILE_COUNT) * 55  # size/1000 BEST FOR FULL DATA SET
INDEX_BUFFER = utils.get_size_directory(DATA_DIR) / 150
FINDEX_MAX_LINES = 100000  # lines updated from 50,000 to 100,000
FINDEX_BUFFER = 10000 # is limited to 10,000 due to it having to be sorted O(nlogn) every time

# RANKING
LINKS_KEY = "links"
BOLDS_KEY = "bolds"
POSITION_KEY = "position"
TFIDF_KEY = "tfidf"
# SEARCH OPTIONS
TEMP_WEIGHT = 1
BOLDS_WEIGHTING = True
POSITIONAL_WEIGHTING = False
PAGERANK_WEIGHTING = True
MD5_DUPLICATE_CHECK = True
USE_SIMHASH = False
BOOLEAN_AND = False
# NOTE: if SIMHASH is enabled, it will replace the MD5 hash check.
LOG = True


def write_bucket(inverted_index_bucket_ref, INDEX_DIR, docID):
    with open(f"{INDEX_DIR}{docID}.index", "w+") as f:
        # {token:{docID:{positions:}}}
        for token in sorted(inverted_index_bucket_ref.keys()):
            f.write(f"{token}~{dict(inverted_index_bucket_ref[token])}\n")
        f.close()


def normalization_term(weighted_vector):
    # sqrt(sum(term_score^2) # term scores == "weighted vector"
    return math.sqrt(sum([term_score * term_score for term_score in weighted_vector]))


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
    if Bard:
        print("================================================")
        print(f"LIBRARY SIZE: {utils.get_size_directory(DATA_DIR) / 1000000}mb")
        print(f"LIBRARY SIZE: {FILE_COUNT} files")
        print(f"INDEX BUFFER SIZE: {INDEX_BUFFER / 1000000}mb")
        print(f"BUFFER FRACTION: {INDEX_BUFFER / utils.get_size_directory(DATA_DIR)}")
        print("================================================")
    print("<--------------MAKING FRAGMENTED INDEX-----------------> 1/2")
    document_store = list()
    inverted_index_bucket = defaultdict(lambda: defaultdict(list))  # {token:{docID:{positions:}}}
    supplemental_info = defaultdict(lambda: defaultdict(list))  # {docID:{links:[], bolds:[]}}
    docID_hashes = defaultdict(int)
    tokenIDs = set()
    docID = 0
    doc_count = utils.count_docs(DATA_DIR)

    # BARD IN!
    if Bard:
        Bard.start("WALKING DATA")
    for subdir, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if ".json" in file:
                # If bucket is at-size, dump bucket
                if sys.getsizeof(inverted_index_bucket) > INDEX_BUFFER:
                    print("\n<---------WRITING BUCKET--------->")
                    write_bucket(inverted_index_bucket, INDEX_DIR, docID)
                    inverted_index_bucket = defaultdict(lambda: defaultdict(list))  # {token:{docID:{positions:}}}
                # Open url-database file
                with open(os.path.join(subdir, file), "r") as f:
                    jsonfile = json.load(f)
                    f.close()
                document_store.append(jsonfile["url"])
                soup = BeautifulSoup(jsonfile["content"], features="lxml")
                tokens = ctoken.tokenize(soup.text)

                # GENERATE TOKEN HASH
                if not USE_SIMHASH:
                    docID_hashes[docID] = hashlib.md5(soup.text.encode('utf-8')).hexdigest()

                # EXTRACT BOLDS
                for bold in soup.find_all(["b", "strong", "h1", "h2", "h3", "title"]):
                    [supplemental_info[docID][BOLDS_KEY].append(word) for word in ctoken.tokenize(bold.text)]

                # EXTRACT LINKS
                # [(link, anchor_text)]
                links = list()
                for link in BeautifulSoup(jsonfile["content"], features="lxml", parse_only=SoupStrainer("a")).find_all("a"):
                    try:
                        if hasattr(link, "href") and validators.url(link["href"]):
                            links.append((link["href"], ctoken.tokenize(link.text)))
                    except KeyError:
                        pass

                supplemental_info[docID][LINKS_KEY] = [link[0] for link in links]
                # TOKENIZE!
                position = 0
                # INDEX ANCHOR TEXT
                for link in links:
                    if link[1]:
                        for token in link[1]:
                            inverted_index_bucket[token][str(docID)].append(position)
                            position += 1
                            tokenIDs.add(token)
                # add rest of tokens
                for token in tokens:
                    inverted_index_bucket[token][str(docID)].append(position)
                    position += 1
                    tokenIDs.add(token)

                # BARD BARDING!

                percent_progress = "{:.5f}".format(docID / doc_count)
                percent_progress_human = "{:.5f}".format((docID / doc_count) * 100)
                if Bard: # and percent_progress % 0.000001 == 0:
                    log = f"{percent_progress_human}% bucket size:{sys.getsizeof(inverted_index_bucket) / 1000000}mb " \
                          f"{subdir}/{file} tokens:{len(tokens)} bolds:{len(supplemental_info[docID][BOLDS_KEY])} " \
                          f"links:{len(links)}"
                    Bard.update(log, replace=True)
                # INCREMENTING DOCID COUNTER
                docID += 1

    print("<---------WRITING FINAL BUCKET--------->")
    write_bucket(inverted_index_bucket, INDEX_DIR, docID)
    inverted_index_bucket.clear()

    print("<---WRITING DOCUMENT STORE------->")
    with open(docID_store_file_filename, "w") as f:
        json.dump(document_store, f)
        f.close()

    print("<-----WRITING TERM ID STORE------>")
    sorted_tokenIDs = sorted(list(tokenIDs))
    with open(termID_map_filename, "w") as f:
        json.dump(sorted_tokenIDs, f)
        f.close()

    print("<-----WRITING SUPPLEMENTAL LINK/BOLD STORE------>")
    with open(supplemental_info_filename, "w") as f:
        json.dump(supplemental_info, f)
        f.close()

    print("<-----WRITING HASH STORE------>")
    with open(docID_hash_filename, "w") as f:
        json.dump(docID_hashes, f)
        f.close()

    print("<-----WRITING INVERTED DOCID MAP------>")
    with open(docID_store_file_filename, "r") as f:
        inverted_docID_map = page_duplicate_util.gen_inverted_docID_map(json.load(f))
        with open(invert_docID_filename, "w") as f:
            json.dump(inverted_docID_map, f)

    print("<-----WRITING INVERTED BOLDS MAP------>")
    utils.make_invert_bolds_term_docID(supplemental_info_filename, inverted_bolds_filename)
    print("<-----WRITING PAGERANK STORE------>")
    utils.make_pagerank_lib(supplemental_info_filename, docID_store_file_filename, pagerank_filename)

    # BARD OUT!
    if Bard:
        Bard.end()


def merge_frag_index():
    """
    All .txt files of fragmented index are opened and a line is read from each. The 'minimum' term/token that
    exists in the read buffer (that which has the lowest index number in the tokenID map) is written.
    :return:
    """
    print(f"FINDEX MAX LINES: {FINDEX_MAX_LINES}")
    print(f"FINDEX BUFFER SIZE: {FINDEX_BUFFER/1000000}mb")
    print("<--------------MERGING FRAGMENTED INDEX-----------------> 2/2")
    # load sorted tokenIDs
    with open(termID_map_filename, "r") as f:
        sorted_tokenIDs = json.load(f)

    if sorted_tokenIDs != sorted(sorted_tokenIDs):
        raise IndexError
    print("termID_map_tokens: ",len(sorted_tokenIDs))

    # BARD IN!
    if Bard:
        Bard.start("MERGING FRAGMENT DATA")

    # open all fragment files for reading
    file_objects = list()
    for subdir, dirs, files in os.walk(INDEX_DIR):
        for file in files:
            if ".index" in file:
                file_objects.append(open(f"{INDEX_DIR}{file}", "r"))

    # IO Token Map and Final Index
    token_map_file = open(token_seek_map_filename, "w")
    # open outfile for final database and the map file
    outfile_ID = 0
    outfile_name = f"{INVERT_DIR}{outfile_ID}.findex"
    outfile = open(outfile_name, "w")
    outfile_lines = 0

    # establish read buffer, which contains terms that have been read but not been written to final index
    read_buffer = defaultdict(lambda: defaultdict(list))  # {token:{docID:[positions]}}
    # establish token map, which contains the seek index in the final output file
    token_map = defaultdict(lambda: defaultdict(list))  # {token:{file:file_seek_position}}
    tokenID = 0

    # corpus token frequency metric
    corpus_token_freq = defaultdict(int) # {token:count}

    while tokenID < len(sorted_tokenIDs):
        # check if outfile is sufficient line size to merit new outfile
        if outfile_lines > FINDEX_MAX_LINES:
            outfile.close()
            print("\n<---------NEW OUTFILE--------->")
            # start new outfile
            outfile_ID += 1
            outfile_name = f"{INVERT_DIR}{outfile_ID}.findex"
            outfile = open(outfile_name, "w")
            outfile_lines = 0
        # iterate through all files and add next line to read buffer
        for file in file_objects:
            for i in range(0, FINDEX_BUFFER):
                line = file.readline().split("~")
                if len(line) != 2:  # [term, positional_data]
                    file_objects.remove(file)
                    break
                token = line[0]
                dictionary = line[1]
                docIDs_to_positions = json.loads(dictionary.replace("'", "\""))  # required for embedded json
                for docID, positions in docIDs_to_positions.items():
                    for position in positions:
                        read_buffer[token][docID].append(position)
                        corpus_token_freq[token] += 1
            Bard.update(f"readbuffer size:{sys.getsizeof(read_buffer) / 1000000}mb\t{file.name}", replace=True)
        sorted_readbuffer_ids = sorted(list(map(lambda x: sorted_tokenIDs.index(x), read_buffer.keys())))
        while len(read_buffer.keys()) > 0:
            if outfile_lines > FINDEX_MAX_LINES:
                outfile.close()
                print("\n<---------NEW OUTFILE--------->")
                # start new outfile
                outfile_ID += 1
                outfile_name = f"{INVERT_DIR}{outfile_ID}.findex"
                outfile = open(outfile_name, "w")
                outfile_lines = 0
            # if sorted_tokenIDs[tokenID] not in read_buffer.keys():
            #     continue
            # find lowest token index of tokens currently in the read buffer
            # get actual token from termID/tokenID
            min_tokenID = sorted_readbuffer_ids.pop(0)
            this_token = sorted_tokenIDs[min_tokenID]
            # gets current cursor position in output file, then writes from read buffer.
            # Notes this outfile position in token map # {token:{file:file_seek_position}}
            outfile_position = outfile.tell()
            token_map[this_token][outfile_name].append(outfile_position)
            outfile.write(f"{dict(read_buffer.pop(this_token))}\n")

            # increment outfile line count
            outfile_lines += 1
            tokenID += 1
            # BARD BARDING!
            percent_progress = float(tokenID / len(sorted_tokenIDs))

            if Bard and round(percent_progress, 3) % 0.001 == 0:
                human = "{:.5f}".format(percent_progress * 100)
                Bard.update(f"{human}%"
                            f"\t{outfile_name}\tRead buffer:{len(read_buffer)}\t"
                            f"{sys.getsizeof(read_buffer) / 1000000}mb\t{outfile_lines}lines\t{this_token}"
                            f"\t{sorted(read_buffer.keys())[0:5]}",
                            replace=True)

    # dump token map which contains {token:{file:file_seek_position}}
    print("<-----WRITING TOKEN SEEK MAP------>")
    json.dump(token_map, token_map_file)
    print("<-----WRITING ------>")
    with open(corpus_token_frequency_filename, "w") as f:
        json.dump(corpus_token_freq, f)
        f.close()

    # close all files
    for file in file_objects:
        file.close()
    outfile.close()
    token_map_file.close()

    # BARD OUT!
    if Bard:
        Bard.end()

def search_multiple(search_queries_ref, token_map_ref, docid_store_ref,
                    findex_dict, duplicate_docIDS, bold_links_map, sorted_pagerank_ref, inverted_bolds):
    ret_results = defaultdict(tuple)
    for query_phrase in search_queries_ref:
        time_search_start = time.perf_counter()
        term_docID_positions = defaultdict(lambda: defaultdict(list)) # {term{docID:[positions]}}
        union = set()  # [docID]
        # QUERY
        tokenized_query = ctoken.tokenize(query_phrase, trim_stopwords=False) # [query_term]
        unique_query_term_freqs = defaultdict(int) # {term:freqs}
        # QUERY TFIDF
        query_weighted_tfidf_dict = defaultdict(float)  # {term:float}
        query_normalized_weighted_tfidf_dict = defaultdict(float)  # {term:float}
        term_idf = defaultdict(float)  # {term:weighted_idf}
        # DCOUMENT TFIDF
        document_normalized_weighted_tf_dict = defaultdict(lambda: defaultdict(float))  # {docID:{term:float}}
        final_doc_scores = defaultdict(float)  # {docID:float}}
        all_stopwords = False
        # FREQ'ING QUERY
        # if the query is extremely long, stop words are removed

        if len(tokenized_query) > 3:
            tokenized_query = [token for token in tokenized_query if token not in set(stopwords.words('english'))]
        unique_keys = set(tokenized_query)
        for toke in tokenized_query:
            unique_query_term_freqs[toke] += 1
        # ACCESSING DOCID's AND POSITIONS
        # accounts for posting information stored in multiple files
        for toke in unique_keys:
            # posting list(s)
            try:
                for file, seek in token_map_ref[toke].items():
                    f = findex_dict[file]
                    for seek_pos in seek:
                        f.seek(seek_pos)  # sends cursor to position
                        posting = orjson.loads(f.readline().replace("'", "\"")) # {docID:[seek_positions]}
                        for docID in posting.keys():
                            # trim off duplicate pages
                            if docID not in duplicate_docIDS and docID not in term_docID_positions[toke]:
                                [term_docID_positions[toke][docID].append(pos) for pos in posting[docID]]
                                union.add(docID)
            except KeyError:
                # removing terms that do not exist in corpus
                del unique_query_term_freqs[toke]
                continue
        file_time2 = time.perf_counter()

        # print("filetime:", (file_time2-file_time1)*1000)
        # QUERY TFIDF VECTOR
        for term in unique_query_term_freqs.keys():
            # weighted term frequency
            weighted_tf = 1 + math.log(unique_query_term_freqs[term])
            # weighted number of documents / number of documents containing term
            weighted_idf = math.log(len(docid_store_ref) / len(term_docID_positions[term]))
            query_tfidf = weighted_tf * weighted_idf
            query_weighted_tfidf_dict[term] = query_tfidf
            term_idf[term] = weighted_idf
        query_n_term = normalization_term(query_weighted_tfidf_dict.values()) # sqrt(sum(query_vector))
        for term in query_weighted_tfidf_dict.keys():
            query_normalized_weighted_tfidf_dict[term] = query_weighted_tfidf_dict[term] / query_n_term


        # BOLDS EXTRACTION
        bolds_docID = set()
        for term in [query for query in query_weighted_tfidf_dict.keys() if query not in set(stopwords.words('english'))]:
            try:
                [bolds_docID.add(docID) for docID in inverted_bolds[term]]
            except IndexError and KeyError:
                pass
        # DOCUMENT PRUNING--------------------------------------------------
        # boolean AND documents
        # boolan AND + bolds
        # boolean AND + bolds + iter(union each set of docIDs for term in reverse idf order)
        # still none? Nuclear option -> union with all docs containing term
        datasets = []
        MINIMUM_ITERS = 1000
        MAXIMUM_ITERS = 2000
        try:
            searchdocs = set.intersection(
                *[set(term_docID_positions[term].keys()) for term in term_docID_positions.keys()])
            datasets.append(f"boolean AND{len(searchdocs)}")
        except TypeError:
            searchdocs = set()
            pass
        predicted_iterations = len(searchdocs) * len(query_normalized_weighted_tfidf_dict.keys())
        while predicted_iterations < MINIMUM_ITERS:
            if all_stopwords:
                break
            counter = 0
            for term, idf in sorted(term_idf.items(), key=lambda x: x[1], reverse=True):
                searchdocs = set.union(*[searchdocs, [docID for docID in term_docID_positions[term]]])
                datasets.append(f"union-termdocs{counter}-{term}-{len(searchdocs)}")
                counter += 1
                predicted_iterations = len(searchdocs)*len(query_normalized_weighted_tfidf_dict.keys())
                if predicted_iterations > MINIMUM_ITERS:
                    break
            if predicted_iterations > MINIMUM_ITERS:
                break
            searchdocs = set.union(*[searchdocs, bolds_docID])
            datasets.append(f"union-bolds{len(searchdocs)}")
            predicted_iterations = len(searchdocs) * len(query_normalized_weighted_tfidf_dict.keys())
            if predicted_iterations > MINIMUM_ITERS:
                break
            searchdocs = set.union(*[searchdocs, union])
            datasets.append(f"union-union{len(searchdocs)}")
            predicted_iterations = len(searchdocs) * len(query_normalized_weighted_tfidf_dict.keys())
            break
        while predicted_iterations > MAXIMUM_ITERS:
            # do some filtering, thats alot of results.
            searchdocs = set.intersection(*[searchdocs, bolds_docID])
            datasets.append(f"intersect-bolds{len(searchdocs)}")
            predicted_iterations = len(searchdocs) * len(query_normalized_weighted_tfidf_dict.keys())
            if predicted_iterations < MAXIMUM_ITERS:
                break
            searchdocs = set.intersection(*[searchdocs, union])
            datasets.append(f"intersect-union{len(searchdocs)}")
            break
        datasets.append(f"iterations:{len(searchdocs) * len(query_normalized_weighted_tfidf_dict.keys())}")

        # DOCUMENT TF-IDF SCORING ---------------------------------
        # COMPUTE DOCUMENTS TF'S
        # term frequency in document, list of score values() for all terms
        for docID in searchdocs:
            for term in unique_query_term_freqs.keys():
                # weighted document tf == 1+log(term frequency in document)
                # document_normalized_weighted_tf_dict[docID].values() == "document vector" or term_scores
                if len(term_docID_positions[term][docID]) == 0:
                    document_normalized_weighted_tf_dict[docID][term] = \
                        1 + math.log(1)
                else:
                    document_normalized_weighted_tf_dict[docID][term] =\
                    1 + math.log(len(term_docID_positions[term][docID]))
            for term in unique_query_term_freqs.keys():
                # NORMALIZE DOCUMENT TF
                n_term = normalization_term(document_normalized_weighted_tf_dict[docID].values())
                document_normalized_weighted_tf_dict[docID][term] =\
                    document_normalized_weighted_tf_dict[docID][term] / n_term

        # COMPUTE FINAL TERM-DOCUMENT RELEVANCE SCORE-----------------------
        # sum(for all terms: query_term_score*document_normalized_weighted_tf_dict)
        non_tfidf_weighting_factor = 1
        for docID in document_normalized_weighted_tf_dict.keys():
            sum_list = list()
            for term in document_normalized_weighted_tf_dict[docID].keys():
                doc_score = query_normalized_weighted_tfidf_dict[term] * document_normalized_weighted_tf_dict[docID][term]
                if doc_score == 0: # its an irrelevant document
                    continue
                if len(document_normalized_weighted_tf_dict[docID].keys()) == 1:  # the query was a single word
                    non_tfidf_weighting_factor = 10
                if BOLDS_WEIGHTING and docID in bolds_docID:
                    doc_score += 0.0001 * non_tfidf_weighting_factor
                    # print("we struck BOLD!")
                if PAGERANK_WEIGHTING:
                    try:
                        page_rank = sorted_pagerank_ref[int(docID)][1]
                        if page_rank > 0.001:
                            doc_score += page_rank*non_tfidf_weighting_factor
                            pass
                    except IndexError:
                        pass
                sum_list.append(doc_score)
            final_doc_scores[docid_store_ref[int(docID)]] = sum(sum_list)

        # SORT AND SLICE RESULTS ----------------------------
        search_results = sorted(final_doc_scores.items(), key=lambda x: x[1], reverse=True)[0:SEARCH_RESULTS]
        time_taken = time.perf_counter() - time_search_start
        # print(tokenized_query, "results", "\n", len(final_doc_scores), "datasets", datasets)
        if union:
            ret_results[query_phrase] = (search_results, time_taken)
        else:
            ret_results[query_phrase] = ([], time_taken)
    return ret_results


if __name__ == '__main__':
    frag_index_exists = True
    all_index_exists = True
    DEBUG = False
    SEARCH_RESULTS = 10
    # Build Indexes
    if not frag_index_exists:
        make_fragment_index()
        frag_index_exists = True

    if not all_index_exists:
        merge_frag_index()
        all_index_exists = True

    print("loading files into memory.....")
    # open FINDEX files
    findex_file_objects = dict()
    for subdir, dirs, files in os.walk(INVERT_DIR):
        for file in files:
            if ".findex" in file:
                f = open(f"{INVERT_DIR}{file}", "r")
                findex_file_objects[f.name] = f
    # add maps back to memory
    with open(token_seek_map_filename, "r") as f:
        from_file_token_map = json.load(f)
    with open(docID_store_file_filename, "r") as f:
        docID_store = json.load(f)
    with open(docID_hash_filename, "r") as f:
        docID_hash_store = json.load(f)
    with open(supplemental_info_filename, "r") as f:
        bolds_links_store = json.load(f)
    with open(corpus_token_frequency_filename, "r") as f:
        corpus_token_frequency = json.load(f)
    with open(invert_docID_filename, "r") as f:
        invert_docID_map = json.load(f)
    with open(inverted_bolds_filename, "r") as f:
        bolds_terms_docIDs = json.load(f)
    with open(pagerank_filename, "r") as f:
        sorted_pagerank = json.load(f)
    duplicate_docIDs = page_duplicate_util.find_duplicates(docID_hash_store)

    # DEBUG ONLY
    if DEBUG:
        print("================================================")
        PRINT_TERMS = 10
        stopwords = [ctoken.tokenize(i)[0] for i in stopwords.words('english')]
        sorted_corpus = [(term, count)
                         for term, count
                         in sorted(corpus_token_frequency.items(), key=lambda x: x[1], reverse=True)
                         if term not in stopwords]
        # avg_freq = sum(list(corpus_token_frequency.values()))/len(corpus_token_frequency)
        # foo = int(avg_freq)-PRINT_TERMS
        # bar = int(avg_freq)+PRINT_TERMS
        print("documents in docID store: ", len(docID_store))
        print("documents in hash store: ", len(docID_hash_store))
        print("duplicates: ", len(duplicate_docIDs))
        print("tokens: ", len(from_file_token_map))
        print(f"avg", len(sorted_corpus))
        avg = int(len(sorted_corpus)/2)
        print(f"median count {PRINT_TERMS} terms: {sorted_corpus[avg:PRINT_TERMS]}")
        print(f"top count {PRINT_TERMS} terms: {sorted_corpus[0:PRINT_TERMS]}")
        print(f"bottom count {PRINT_TERMS} terms: {list(reversed(sorted_corpus))[0:PRINT_TERMS]}")
        top_pageranks = sorted_pagerank[0:PRINT_TERMS]
        bottom_pageranks = list(reversed(sorted_pagerank))[0:PRINT_TERMS]
        print(f"top pagerank {PRINT_TERMS} terms: {top_pageranks}")
        print(f"bottom pagerank {PRINT_TERMS} terms: {bottom_pageranks}")
        print("================================================")
        print()

    # Search
    search_queries = ["ai club", "master of software engineering", "MOSFET", "Dingo ate me baby", "support document",
                      "browser", "the university of california irvine ai club workshop", "sourcer","lawks", "lawler",
                      "breast cancer wisconsin", "computer science", "informatics", "rwxrwxrwx", "laveman",
                      "language for distributed embedded systems", "a",
                      "krisberg org", "kovarik@mcmail.cis.mcmaster.ca", "cbcl", ]
    search_queries = ["language for distributed embedded systems",]
    results = search_multiple(search_queries, from_file_token_map, docID_store,
                              findex_file_objects, duplicate_docIDs, bolds_links_store,
                              sorted_pagerank, bolds_terms_docIDs)
    for query in results.keys():
        result = results[query][0]
        time_taken = results[query][1]
        print(f"Query '{query}' took {time_taken*1000} milliseconds.")
        for index, url in enumerate(result):
            print(f"{index + 1}. {url}")
        print()