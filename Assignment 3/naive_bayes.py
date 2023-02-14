import numpy as np
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CATEGORIES = {"sport": 1, "business": 2, "politics": 3, "entertainment": 4, "tech": 5}


class NaiveBayes:
    def __init__(self, mode: tuple = (1, 1), remove_stopwords: bool = False):
        self.mode = mode
        self.remove_stopwords = remove_stopwords
        self.category_dist = {}
        self.probability_dist = {}
        self.vectorized_dict = {}
        self.columns = []

    def fit(self, X, Y):
        self.vectorized_dict, self.columns = self.vectorize(X)
        self.prior(Y)
        self.like_hood(Y, 1)

    def predict(self, X):
        predictions = []

        vector, columns = self.vectorize(X)

        for i in range(len(vector)):
            probabilities = {1: self.category_dist[1],
                             2: self.category_dist[2],
                             3: self.category_dist[3],
                             4: self.category_dist[4],
                             5: self.category_dist[5]}

            for j in range(len(vector[i])):
                if vector[i][j] == 0:
                    continue
                elif "%s|1" % columns[j] in self.probability_dist.keys():
                    probabilities[1] += vector[i][j] * self.probability_dist["%s|1" % columns[j]]
                    probabilities[2] += vector[i][j] * self.probability_dist["%s|2" % columns[j]]
                    probabilities[3] += vector[i][j] * self.probability_dist["%s|3" % columns[j]]
                    probabilities[4] += vector[i][j] * self.probability_dist["%s|4" % columns[j]]
                    probabilities[5] += vector[i][j] * self.probability_dist["%s|5" % columns[j]]

            predictions.append(max(probabilities.items(), key=operator.itemgetter(1))[0])

        return predictions

    def prior(self, Y):
        categories = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for category in Y:
            categories[category] += 1

        for category in categories.keys():
            self.category_dist[category] = np.log(categories[category] / len(Y))

    def vectorize(self, corpus):
        vectorizer = CountVectorizer(ngram_range=self.mode)
        if self.remove_stopwords:
            vectorizer = CountVectorizer(ngram_range=self.mode, stop_words=ENGLISH_STOP_WORDS)

        transformed = vectorizer.fit_transform(corpus)

        return transformed.toarray(), list(vectorizer.get_feature_names_out())

    def like_hood(self, Y, alpha):
        total_count = len(self.columns)

        sport_count = np.sum(self.vectorized_dict[Y == 1])
        business_count = np.sum(self.vectorized_dict[Y == 2])
        politics_count = np.sum(self.vectorized_dict[Y == 3])
        entertainment_count = np.sum(self.vectorized_dict[Y == 4])
        tech_count = np.sum(self.vectorized_dict[Y == 5])

        sport_rows = np.sum(self.vectorized_dict[Y == 1], axis=0)
        business_rows = np.sum(self.vectorized_dict[Y == 2], axis=0)
        politics_rows = np.sum(self.vectorized_dict[Y == 3], axis=0)
        entertainment_rows = np.sum(self.vectorized_dict[Y == 4], axis=0)
        tech_rows = np.sum(self.vectorized_dict[Y == 5], axis=0)

        for i in range(len(self.columns)):
            self.probability_dist["%s|1" % (self.columns[i])] = np.log((sport_rows[i] + alpha) / (total_count + sport_count))
            self.probability_dist["%s|2" % (self.columns[i])] = np.log((business_rows[i] + alpha) / (total_count + business_count))
            self.probability_dist["%s|3" % (self.columns[i])] = np.log((politics_rows[i] + alpha) / (total_count + politics_count))
            self.probability_dist["%s|4" % (self.columns[i])] = np.log((entertainment_rows[i] + alpha) / (total_count + entertainment_count))
            self.probability_dist["%s|5" % (self.columns[i])] = np.log((tech_rows[i] + alpha) / (total_count + tech_count))
