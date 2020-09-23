from classifier import Classifier
from classifier_kmeans import KMeansClassifier
from classifier_rulebased import RuleBasedClassifier

import random
from sklearn import metrics
import numpy as np


class HybridClassifier(Classifier):

    def __init__(self, mode=0):
        super().__init__(None)
        self.mode = mode

    def train(self, training_data: dict, params: dict):
        self.words = [["ewhor", "e-whor"],
                      ["stresser", "booter"], [" rat "], ["crypt", "fud"]]
        self.kmeans = training_data['kmeans']

    def pred(self, example):
        if self.mode:
            distances = self.kmeans.pred_full(example)
            minimum = self.kmeans.labels[distances.index(min(distances))]
            sortd = np.argsort(distances)
            if distances[sortd[1]] - distances[sortd[0]] < 0.01:
                max_mentions = 0
                max_class = []
                for classification in range(len(self.words)):
                    mentions = 0
                    for word in self.words[classification]:
                        if word in example:
                            mentions += 1
                    if mentions >= max_mentions:
                        max_mentions = mentions
                        max_class.append(classification)
                if len(max_class) == 1:
                    return max_class[0]
                else:
                    return minimum
            else:
                return minimum
        else:
            max_mentions = 0
            max_class = []
            for classification in range(len(self.words)):
                mentions = 0
                for word in self.words[classification]:
                    if word in example:
                        mentions += 1
                if mentions >= max_mentions:
                    max_mentions = mentions
                    max_class.append(classification)
            if len(max_class) == 1:
                return max_class[0]
            else:
                return self.kmeans.pred(example)

    def test(self, data):
        super().test(data)
        self.present_results()
