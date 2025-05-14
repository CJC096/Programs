import re
import numpy as np
from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(sentence):
    # Use a more robust tokenizer
    words = re.findall(r'\b\w+\b', sentence)
    return words


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
