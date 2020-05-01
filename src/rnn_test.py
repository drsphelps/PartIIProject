from rnn import RNNClassifer
from get_training import crime_dataset, noncrime_dataset


from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


class RNNTest():
    def __init__(self):
        self.word_embedding = Word2Vec.load("data/models/fasttext.modelFile")
        self.RNN = RNNClassifer(self.word_embedding)
        self.load = True
        self.__split_dataset()

    def __split_dataset(self, size=0.2):
        crime = crime_dataset(True)
        noncrime = noncrime_dataset(3000, True, self.load, True)

        remainder, self.crime_training = train_test_split(
            crime, test_size=1.-size)
        self.crime_val, self.crime_test = train_test_split(
            remainder, test_size=0.5)

        remainder, self.noncrime_training = train_test_split(
            crime, test_size=1.-size)
        self.noncrime_val, self.noncrime_test = train_test_split(
            remainder, test_size=0.5)

    def validate(self):
        training = {
            "crime": self.crime_training,
            "noncrime": self.noncrime_training
        }
        params = {
            "embedding_dim": 120,
            "max_words": 100000,
            "max_length": 200,
        }
        self.RNN.train(training, params)
        self.RNN.test(self.crime_val + self.noncrime_val)

    def test(self):
        self.RNN.test(self.crime_test + self.noncrime_test)
