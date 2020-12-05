import hashlib
from nltk.corpus import stopwords
import ctoken
from collections import defaultdict
stopwords = [ctoken.tokenize(i)[0] for i in stopwords.words('english')]
HASHBITS = 128

def hash(words):
    """
    Creates a hashcode using the simhash method, and md5 128 bit hashcodes.
    :param words: list of tokenized words
    :return: 128 bit hash code
    """
    # compute word frequencies
    if not words:
        raise IndexError
    freqs = defaultdict(int)
    for word in words:
        freqs[word] += 1
    # computes the binary of set(words) and outputs a (word_binary, word_frequency) tuple, removing stopwords
    # zfill is for extra 0's in the binary to get to HASHBITS
    binaries = [(bin(int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16))[2:].zfill(HASHBITS), freqs[word])
                for word in set(words) if word not in stopwords]
    # computes the simhash converted word binary, which converts 0's to -1's and multiplies by the word weight
    simhash_bin_arrays= [[-1 * weight if bit =="0" else weight for bit in binary] for binary, weight in binaries]
    # sums the columns into an array as long as the size of bits in previous
    try:
        sum_array = [sum([simhash_bin_arrays[row][col]
                      for row in range(0, len(simhash_bin_arrays))]) for col in range(0, len(simhash_bin_arrays[0]))]
    except IndexError:
        print(binaries)
        print(simhash_bin_arrays)
        print(len(simhash_bin_arrays))
    # converts sums back to binary representation
    final_list = ["1" if val > 0 else "0" for val in sum_array]
    # convert to string
    doc_simhash_hash = "".join(final_list)
    return doc_simhash_hash


def hamming_distance(hash1, hash2):
    # if the bits do not match, increment hamming distance count
    return sum([1 if elem1 != elem2 else 0 for elem1, elem2 in zip(hash1, hash2)])


def find_similar_to_doc(main_docID, hashdict, distance):
    ret = list()
    for docID, hash in hashdict.items():
        distance = hamming_distance(hashdict[main_docID], hash)
        if distance < distance:
            ret.append(docID, distance)
    return ret


if __name__ == '__main__':
    words = ["dingo", "ate", "me", "baby", ]
    words6 = ["dingo", "ate", "me", "baby", ]
    words2 = ["dingo", "ate", "me", "baby", ]
    print(hamming_distance(hash(words), hash(words2)))