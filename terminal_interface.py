import sys
import main
import json

INDEX_DIR = "INDEX/"
INVERT_DIR = "INVERT/"
final_outfile_name = f"{INVERT_DIR}final.txt"
token_map_file_name = f"{INVERT_DIR}token_map.txt"
termID_map_name = f"{INDEX_DIR}termID_map.json"
docID_store_file_name = f"{INDEX_DIR}docid.map"


with open(token_map_file_name, "r") as f:
    token_map = json.load(f)
with open(docID_store_file_name, "r") as f:
    docID_store = json.load(f)

if len(sys.argv[1:]) > 0:
    search_terms = sys.argv[1:]
    results = main.search_multiple(search_terms, token_map_ref=token_map, docid_store_ref=docID_store)

    for query in results.keys():
        result = results[query][0]
        time_taken = results[query][1]
        print(f"Query '{query}' took {time_taken} seconds.")
        for index, url in enumerate(result):
            print(f"{index+1}. {url}")
        print()
else:
    while True:
        results = main.search_multiple([input("Please input query:")], token_map_ref=token_map, docid_store_ref=docID_store)
        for query in results.keys():
            result = results[query][0]
            time_taken = results[query][1]
            print(f"Query '{query}' took {time_taken} seconds.")
            for index, url in enumerate(result):
                print(f"{index + 1}. {url}")
            print()