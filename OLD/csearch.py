import ctoken
import loadbard
from collections import defaultdict
from collections import OrderedDict

# inverted index {word: {docID: {positions:, bold_bool:}}
BOLD_MULTI = 2

def search(search_phrase, invert_ref) -> list:
    document_results = list()
    for token in ctoken.tokenize(search_phrase):
        token_dict = defaultdict(dict)
        for docID in invert_ref[token].keys():
            token_dict[token][docID] = {
                "bold_bool": invert_ref[token][docID]["bold_bool"],
                "count": len(invert_ref[token][docID]["positions"]),
                #"tfidf": gen_tfidf(word=token, docID=docID, invert_ref=invert_ref),
            }
        document_results.append(token_dict)
    return rank(document_results)

def dict_key_intersection(lst1, lst2):
    return [value for value in lst1.items() if value[0] in lst2.keys()]

def rank(struct): # {term:{document:{bold_bool:, count:}}}
    # scores
    meta = list()
    for term_struct in struct:
        term_ranking = defaultdict(int)  # {docID: score}}
        for term in term_struct.keys():
            for docID in term_struct[term]:
                if term_struct[term][docID]["bold_bool"]:
                    term_ranking[docID] = term_struct[term][docID]["count"] * BOLD_MULTI
                else:
                    term_ranking[docID] = term_struct[term][docID]["count"]
        meta.append(term_ranking)
    # intersect
    if len(meta) > 1:
        ranked_intersect_pairs = sorted(dict_key_intersection(meta[0], meta[1]), key=lambda x: x[1], reverse=True)
    else:
        ranked_intersect_pairs = sorted(meta[0].items(), key=lambda x: x[1], reverse=True)
    return ranked_intersect_pairs
    # sorted(ranking.items(), key=lambda x: x[1], reverse=True)