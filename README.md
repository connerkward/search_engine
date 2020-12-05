TO OPERATE:

First install the required packages and build the INDEX. Folders "INDEX" and "INVERT" must be created in project root prior to this.

MAC AND LINUX 
(for Windows use approprite directory making command)
~~~
mkdir INVERT
mkdir INDEX
pip install requirements.txt
python make_index.py
~~~

This will take a while, so grab a snack.

After this is complete, you will be able to search for terms. 

to search for term via the command line:
~~~
python terminal_interface.py
~~~
You can also provide queries directly in the command, but results are slow due to the opening of files each time the program is run.
~~~
python terminal_interface.py "your query" "your second query"
~~~
or use the web interface
~~~
python web_interface.py
~~~

