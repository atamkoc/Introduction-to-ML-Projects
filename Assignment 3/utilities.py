import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer


def calculate_accuracy(y, predictions):
    accuracy = np.sum(y == predictions) / len(y)
    return accuracy


def generate_ngrams(text, n_gram=1, stop=False):
    stop = set(ENGLISH_STOP_WORDS) if stop else {}

    tokens = []
    for word in text.lower().split(" "):
        if word != "" and word not in stop:
            tokens.append(word)

    z = []
    for index in range(len(tokens)):
        if index + n_gram > len(tokens):
            break
        z.append(tokens[index:index + n_gram])

    n_grams = []
    for text in z:
        n_grams.append(" ".join(text))
    return n_grams


def calculate_freq(text_list, category_list):
    dict_ngram = {"general": {}}
    for category in category_list:
        dict_ngram[category] = {}

    for index in range(len(text_list)):
        for word in generate_ngrams(text_list[index], 1):
            if word not in dict_ngram[category_list[index]]:
                dict_ngram[category_list[index]][word] = 1
                dict_ngram["general"][word] = 1
            else:
                dict_ngram[category_list[index]][word] += 1
                dict_ngram["general"][word] += 1
    return dict_ngram


def get_ngrams_and_count(text, ngrams=(1, 1), nr=None):
    vec = CountVectorizer(ngram_range=ngrams).fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    return words_freq


def calculate_accuracy(Y, predictions):
    count = 0
    for index in range(len(Y)):
        if Y[index] == predictions[index]:
            count += 1

    print(f"Accuracy is {count / len(Y)}")
