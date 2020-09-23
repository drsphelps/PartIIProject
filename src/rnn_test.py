from classifier_rnn import RNNClassifer
from utils.get_training import crime_dataset, noncrime_dataset


from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


class RNNTest():
    def __init__(self):
        self.word_embedding = Word2Vec.load("data/models/fasttext.modelFile")
        self.RNN = RNNClassifer(self.word_embedding)
        self.load = True
        self.__split_dataset()

    def __split_dataset(self, size=0.1):
        crime = crime_dataset(True)
        noncrime = noncrime_dataset(3000, True, self.load, True)

        remainder, self.crime_test = train_test_split(
            crime, test_size=size)
        self.crime_training, self.crime_val = train_test_split(
            remainder, test_size=size/(1.-size), shuffle=True)

        remainder, self.noncrime_test = train_test_split(
            noncrime, test_size=size)
        self.noncrime_training, self.noncrime_val = train_test_split(
            remainder, test_size=size/(1.-size), shuffle=True)

    def validate(self):
        training = {
            "crime": self.crime_training,
            "noncrime": self.noncrime_training,
            "crime_val": self.crime_val,
            "noncrime_val": self.noncrime_val,
            "all": [self.crime_test, self.crime_training, self.crime_val,
                    self.noncrime_test, self.noncrime_training, self.noncrime_val]
        }
        params = {
            "embedding_dim": 100,
            "max_words": 100000,
            "max_length": 200,
            "epochs": 10
        }
        self.RNN.train(training, params)
        self.RNN.test(np.concatenate(
            (self.crime_val, self.noncrime_val), axis=0))

    def test(self):
        tests = np.concatenate((self.crime_test, self.noncrime_test), axis=0)
        self.RNN.test(tests)
        crime = []
        noncrime = []
        for d in range(len(self.RNN.predictions)):
            if self.RNN.predictions[d] == 1:
                crime.append(tests[d])
            else:
                noncrime.append(
                    (self.RNN.groundtruths[d], self.RNN.predictions[d]))
        with open("data/crime.data", "wb") as f:
            pickle.dump(crime, f)
        with open("data/noncrime.data", "wb") as f:
            pickle.dump(noncrime, f)


rnntest = RNNTest()
rnntest.validate()
# rnntest.test()
