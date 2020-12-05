import re
from collections import defaultdict
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# @ and $ are excluded from TABLE because they add important context to numbers and email addresses, respectively
# replace_string = "\n\t\r^_`{|}~?<=>*+#%!;:/[]\'\",()_”“‘’\\"
replace_string = "\n\t\r^_`{|}~?<=>*+#%!;:/[]\'\",()_”“‘’\\&$"
TABLE2 = str.maketrans(replace_string, " "*len(replace_string))
TABLE_NOSPACE = str.maketrans("", "", "-")
stemmer = SnowballStemmer("english")
STOPWORDS = set(stopwords.words('english'))
NO_NUMERIC = True


def tokenize(line: str, trim_stopwords=False) -> list:
    tokenlist = list()
    line = re.sub(r"\. ", ' ', line) # strips periods at the end of sentences
    for word in line.lower().translate(TABLE2).split():
        # lower:O(N) + translate:O(N) + split:O(N) + forloop:O(N) = O(4N) = O(N)
        try:
            word = word.strip(".@").translate(TABLE_NOSPACE)
            if trim_stopwords and word in STOPWORDS:
                break
            if word != '':
                if NO_NUMERIC and any(map(str.isdigit, word)):
                    pass
                else:

                    tokenlist.append(stemmer.stem(word).encode("ascii").decode())
        except UnicodeEncodeError:  # for handling bad characters in word
            pass
    return tokenlist

def computeWordFrequencies(tokenlist: list) -> dict:
    # O(N) Time Complexity, because worst case each word in for loop is visited once.
    tokencount = defaultdict(int)  # str: int
    if tokenlist and tokenlist is not None:
        for word in tokenlist:
            tokencount[word] += 1
    return tokencount

def computeWordFrequencies_postion_old(tokenlist: list) -> dict:
    # O(N) Time Complexity, because worst case each word in for loop is visited once.
    tokencount = defaultdict(set)  # {word: [positions]} # len(positions) doubles as count
    if tokenlist is not None:
        for word in tokenlist:
            for i, x in enumerate(tokenlist):
                if x == word:
                    tokencount[word].add(i)
    return tokencount

def computeWordFrequencies_postion_linear(tokenlist: list) -> dict:
    # O(N) Time Complexity, because worst case each word in for loop is visited once.
    tokencount = defaultdict(list)  # {word: [positions]} # len(positions) doubles as count
    if tokenlist is not None:
        index = 0
        for word in tokenlist:
            tokencount[word].append(index)
            index += 1
    return tokencount

if __name__ == '__main__':
    stri= "master of software engineering"
    print(tokenize(stri))