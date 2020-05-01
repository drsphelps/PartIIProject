from kmeans import KMeansClassifier
from rulebased import RuleBasedClassifier
from hybrid import HybridClassifier
from get_training import collect_unlabelled_data, load_crime_data
from train_tfidf import TFIDF

from sklearn.model_selection import train_test_split
from gensim.models import TfidfModel, Doc2Vec
from gensim.corpora import Dictionary
import pickle
import numpy as np


class CrimeTypeTest():
    def __init__(self):
        self.load = False
        self.tfidf = TFIDF.load("data/models/tfidf.modelFile")
        self.doc2vec = Doc2Vec.load("data/models/word2vec.modelFile")
        self.rulebased = RuleBasedClassifier()
        self.hybrid = HybridClassifier()
        self.hybrid.train(None, None)
        self.number = 0

    def __split_dataset(self):

        # Get labelled data
        labelled_e = load_crime_data("ewhore", 0)
        labelled_s = load_crime_data("stresser", 1)
        labelled_r = load_crime_data("rat", 2)
        labelled_c = load_crime_data("crypter", 3)

        # Collect or load unlabelled data
        if self.load:
            with open("cluster.data", "rb") as f:
                self.U_train = pickle.load(f)
        else:
            unlabelled_e = collect_unlabelled_data(170, self.number)
            unlabelled_s = collect_unlabelled_data(92, self.number)
            unlabelled_r = collect_unlabelled_data(114, self.number)
            unlabelled_c = collect_unlabelled_data(299, self.number)
            self.U_train = unlabelled_e + unlabelled_s + unlabelled_r + unlabelled_c
            with open("cluster.data", "wb") as f:
                pickle.dump(self.U_train, f)

        rem, X_e = train_test_split(
            labelled_e, test_size=0.9, shuffle=True)
        X_et, X_ev = train_test_split(
            rem, test_size=0.5, shuffle=True)
        rem, X_s = train_test_split(
            labelled_s, test_size=0.9, shuffle=True)
        X_st, X_sv = train_test_split(
            rem, test_size=0.5, shuffle=True)
        rem, X_r = train_test_split(
            labelled_r, test_size=0.9, shuffle=True)
        X_rt, X_rv = train_test_split(
            rem, test_size=0.5, shuffle=True)
        rem, X_c = train_test_split(
            labelled_c, test_size=0.9, shuffle=True)
        X_ct, X_cv = train_test_split(
            rem, test_size=0.5, shuffle=True)

        self.X_train = X_e + X_s + X_r + X_c
        self.X_val = X_ev + X_sv + X_rv + X_cv
        self.X_test = X_et + X_st + X_rt + X_ct

    def validate_kmeans(self, embedding="d2v", number=500):
        self.number = number
        if embedding == "tfidf":
            self.kmeans = KMeansClassifier(self.tfidf)
        else:
            self.kmeans = KMeansClassifier(self.doc2vec)

        training = {
            "X_train": self.X_train,
            "U_train": self.U_train
        }
        params = {
            "k": 4,
            "metric": (lambda x, y: np.linalg.norm(x-y)),
            "tolerance": 0.0001,
            "max_iterations": 500
        }

        self.kmeans.train(training, params)
        self.kmeans.test(self.X_val)

    def test_kmeans(self):
        self.kmeans.test(self.X_test)

    def test_rb(self):
        self.rulebased.test(self.X_test)

    def test_hybrid(self):
        self.hybrid.test(self.X_test)
