# Search Engine in Python

This is a "search engine" built in python that uses term frequency–inverse document frequency (TF-IDF) and pagerank to sort results. Indexes creation includes duplicate removal hashing, ntlk tokenization with stopword removal.

TO OPERATE:

First install the required packages. Working index is inlcuded in files, however, to rebuild the index on a new set of files, folders "INDEX", "INVERT", and "DATA" must be created in project root prior to this. Depending on OS, make sure the variable DATA_DIR points to the right folder. This program was run on Windows. To build the index, inside the DATA folder place folders with website names and inside the website files. 


MAC AND LINUX 
(on Windows use approprite directory making command)
~~~
mkdir INVERT
mkdir INDEX
mkdir DATA
pip install requirements.txt
python make_index.py
~~~

This will take a while, so grab a snack.

After this is complete, you will be able to search for terms. 

to search for a term via the command line:
~~~
python terminal_interface.py
~~~
You can also provide queries directly in the command, but results are slow due to the opening of files each time the program is run.
~~~
python terminal_interface.py "your query" "your second query"
~~~
or use the web interface, which is faster because it keeps the index file open.
~~~
python web_interface.py
~~~

