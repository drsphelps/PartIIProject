from classifier import Classifier
from kmeans import KMeansClassifier
from rulebased import RuleBasedClassifier

import random


class HybridClassifier(Classifier):

    def __init__(self):
        super().__init__(None)

    def train(self, training_data: dict, params: dict):
        self.words = [["ewhor", "e-whor"],
                      ["stresser", "booter"], [" rat "], ["crypt", "fud"]]
        self.kmeans = KMeansClassifier.load("data/models/kmeans.modelFile")

    def pred(self, example: list):
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
