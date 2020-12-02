from flask import Flask, request, render_template
import main
import json

app = Flask(__name__)

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

template = '<form method="POST">\n\t<h1>SEARCH</h1>\n\t<input name="search">\n\t<input type="submit">\n</form>'

@app.route('/')
def my_form():
    return template

@app.route('/', methods=['POST'])
def my_form_post():
    query = request.form['search']
    results = main.search_multiple([query], token_map_ref=token_map, docid_store_ref=docID_store)
    result_str = list()
    for query in results.keys():
        result = results[query][0]
        time_taken = results[query][1]
        result_str.append(f"Query '{query}' took {time_taken} seconds.<br>")
        for index, url in enumerate(result):
            result_str.append(f"{index + 1}. {url}<br>")
        result_str.append("<br>")
    result_str = "".join(result_str)
    return template + result_str
