import index
import math
# CONSTANTS
DATA_DIR = "DATA"

INDEX_DIR = "INDEX/"
INVERT_DIR = "INVERT/"
BUCKET_SIZE = 200
'''
"artifici": {
    "0": {
        "positions": [
                0,
                22,
                45,
                211,
                381
        ],
    "bold_bool": true
    },
}
'''


docID = 0
word = "artifici"

print(index.load_single_doc_from_index_file(docID, BUCKET_SIZE)[str(docID)]["positions"][word])



