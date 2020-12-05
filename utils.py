import os
import json


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