from classifier import Classifier

import random


class RuleBasedClassifier(Classifier):

    def __init__(self):
        super().__init__(None)
        self.words = [["ewhor", "e-whor"],
                      ["stresser", "booter"], [" rat "], ["crypt", "fud"]]

    def train(self, training_data: dict, params: dict):
        pass

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
            return random.choice(range(0, len(self.words)))
