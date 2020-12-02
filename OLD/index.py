import json
import ctoken
import os
from bs4 import BeautifulSoup
import lxml # needed
from collections import OrderedDict
from nltk.stem import PorterStemmer
from loadbard import LoadBard
import math

# TOOLING
stemmer = PorterStemmer() # stemmer.stem()
tokenizer = ctoken.tokenize
load_bard = False
load_bard = LoadBard()
LB_USE_TIMER = True
LB_DISPLAY_TYPE = "percentage"  # can be wheel, bar, or detail
BOLD_TAGS = ["title", "h1", "h2", "h3", 'b', "strong",]

# CONSTANTS
TABS = 0  # defaults to 4, 0 to save size

def check_bolds(word, bolds):
    if word in bolds:
        return True
    else:
        return False

def index_entry_freq_only(wordcount_index_ref, docID, soup):
    wordcount_index_ref[docID] = ctoken.computeWordFrequencies(tokenizer(soup.text))

def index_entry(wordcount_index_ref, docID, soup):
    token_soup = tokenizer(soup.text)
    bold_tokens = list()
    for bold_tag in BOLD_TAGS:
        for elem in soup.find_all(bold_tag):
            [bold_tokens.append(i) for i in tokenizer(line=elem.text, trim_stopwords=True)]
    wordcount_index_ref[docID] = {
                                "bolds": bold_tokens,
                                # "bold_positions": [token_soup.index(item) for item in bold_tokens],
                                "positions": ctoken.computeWordFrequencies_postion_linear(token_soup),
                                }

def make_index(wordcount_index_ref, document_store, abs_decrementer, abs_docs, INDEX_TO_FILE, DATA_DIR, INVERT_EXPORT_DIR, INDEX_CHUNK_SIZE, tokenizer=tokenizer):
    if load_bard:
        load_bard.start("BUILDING INDEX", timer=LB_USE_TIMER, type=LB_DISPLAY_TYPE)
    for subdir, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if ".json" in file:
                with open(os.path.join(subdir, file)) as f:
                    jsonfile = json.load(f)
                soup = BeautifulSoup(jsonfile["content"], features="lxml")
                document_store.append(jsonfile["url"])
                docID = document_store.index(jsonfile["url"])

                index_entry(wordcount_index_ref=wordcount_index_ref, docID=docID, soup=soup)

                abs_decrementer -= 1
                if INDEX_TO_FILE and len(wordcount_index_ref) == INDEX_CHUNK_SIZE:
                    archive_index(wordcount_index_ref=wordcount_index_ref, EXPORT_DIR=INVERT_EXPORT_DIR)
                    wordcount_index_ref = dict()
                elif INDEX_TO_FILE and abs_decrementer < INDEX_CHUNK_SIZE and abs_decrementer == 0:
                    archive_index(wordcount_index_ref=wordcount_index_ref, EXPORT_DIR=INVERT_EXPORT_DIR)
                    wordcount_index_ref = dict()

                if load_bard and len(wordcount_index_ref) != 0: # len(wordcount_index_ref) % INDEX_CHUNK_SIZE == 0
                    thing = float("{:.2f}".format((int(docID)/abs_docs)*100))
                    load_bard.update(f"bucket size: {len(wordcount_index_ref)}\t{thing}%\t{file}", replace=True)
    wordcount_index_ref.clear()
    with open("INDEX2/document_look_up_table.json", "w") as f:
        json.dump(document_store, f, indent=TABS)
    if load_bard:
        load_bard.end()

def archive_index(wordcount_index_ref, EXPORT_DIR):
    with open(EXPORT_DIR + str(min(wordcount_index_ref.keys())) + ".json", 'w', encoding='utf-8') as f:
        export_dict = {key: wordcount_index_ref[key] for key in sorted(wordcount_index_ref.keys())}
        json.dump(export_dict, f, indent=TABS)
    if len(wordcount_index_ref.keys()) != len(export_dict):
        raise ValueError

# inverted index {word: {docID: {positions:, bold_bool:}}

def get_token_count_corpus(word, invert_ref):
    token_count = 0
    for docID in invert_ref[word].keys():
        token_count += len(invert_ref[word][docID]["positions"])
    return token_count

def get_token_count_document(word, docID, invert_ref):
    if docID in invert_ref[word]:
        return len(invert_ref[word][docID]["positions"])
    return 0

def gen_tfidf(word, docID, invert_ref):
    return get_token_count_document(word, docID, invert_ref)/get_token_count_corpus(word, invert_ref)

def update_invert_with_tfidf(invert_ref):
    if load_bard:
        load_bard.start("ADDING TFIDF TO INDEX", timer=LB_USE_TIMER, type=LB_DISPLAY_TYPE)
        count = 0
    for word in invert_ref.keys():
        for docID in invert_ref[word].keys():
            invert_ref[word][docID]["tfidf"] = gen_tfidf(word, docID, invert_ref)
        if load_bard:
            percent_progress = float("{:.2f}".format(count/len(invert_ref.keys())))
            load_bard.update(f"{percent_progress*100}%", replace=True)
            count += 1
    if load_bard:
        load_bard.end()

def invert_index(inverted_index_ref, wordcount_index, tfidf=False):
    if load_bard:
        load_bard.start("INVERTING INDEX", timer=LB_USE_TIMER, type=LB_DISPLAY_TYPE)
    for docID in [str(i) for i in sorted([int(i) for i in wordcount_index.keys()])]:
        for word in wordcount_index[docID]["positions"]:
            if word in inverted_index_ref.keys():
                inverted_index_ref[word][docID] = {
                    "positions": wordcount_index[docID]["positions"][word],
                    "bold_bool": check_bolds(word, wordcount_index[docID]["bolds"]),
                }
            else:
                inverted_index_ref[word] = {
                    docID: {
                        "positions": wordcount_index[docID]["positions"][word],
                        "bold_bool": check_bolds(word, wordcount_index[docID]["bolds"]),
                }
                }

        percent_progress = float("{:.2f}".format(int(docID)/len(wordcount_index.keys())))
        if load_bard and percent_progress % 0.01 == 0:
            load_bard.update(f"{percent_progress*100}%", replace=True)
    if load_bard:
        load_bard.end()
    if tfidf:
        update_invert_with_tfidf(invert_ref=inverted_index_ref)

def invert_index_from_file(inverted_index_ref, INDEX_DIR, tfidf=False):
    if load_bard:
        load_bard.start("LOADING INDEX FILES", timer=LB_USE_TIMER, type=LB_DISPLAY_TYPE)
    # Load entire wordcount index into memory
    wordcount_index_ref = dict()
    for subdir, dirs, files in os.walk(INDEX_DIR):
        files_read = 0
        for file in files:
            with open(INDEX_DIR + file) as f:
                json_obj = json.load(f)
                for docID in json_obj.keys():
                    wordcount_index_ref[docID] = json_obj[docID]
                percent_progress = float("{:.2f}".format(files_read / len(files)))
                if load_bard and percent_progress % 0.001 == 0:
                    load_bard.update(f"{percent_progress * 100}%", replace=True)
                files_read += 1
    if load_bard:
        load_bard.end()
    # invert in-memory index
    invert_index(inverted_index_ref, wordcount_index_ref, tfidf=tfidf)

# INVERT INDEX TO FILE
def invert_from_to_file_simple(wordcount_index_ref, inverted_index_ref,
                               INDEX_DIR, INVERT_EXPORT_DIR, INVERT_TO_FILE, INVERT_BUCKET_SIZE, tfidf=False):
    #index {docID:{ { bolds:[bold words] word_dict:{} } }}
    #inverted index {word: {docID: {word count in document, bold_bool}}

    invert_index_from_file(
        inverted_index_ref=inverted_index_ref,
        INDEX_DIR=INDEX_DIR,
        tfidf=tfidf,
    )
    # how to bucket-ize inverted index
    if INVERT_TO_FILE:
        with open(INVERT_EXPORT_DIR + "invert" + ".json", 'w', encoding='utf-8') as f:
            json.dump(inverted_index_ref, f, indent=TABS)


def load_invert(INVERT_DIR, inverted_index_ref):
    with open(INVERT_DIR + "invert.json") as f:
        return json.load(f)


def load_doc_store():
    with open("INDEX2/document_look_up_table.json", "r") as f:
        return list(json.load(f))

def load_word_store():
    with open("word_look_up_table.json", "r") as f:
        return json.load(f)

def load_single_doc_from_index_file(docID, INDEX_CHUNK_SIZE):
    with open(f"INDEX/{find_bucket(docID, INDEX_CHUNK_SIZE)}.json", "r") as f:
        return json.load(f)

def find_bucket(docID, bucket_size):
    return int(math.floor(docID / bucket_size)) * bucket_size