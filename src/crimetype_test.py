from classifier_kmeans import KMeansClassifier
from classifier_rulebased import RuleBasedClassifier
from classifier_hybrid import HybridClassifier
from utils.get_training import collect_unlabelled_data, load_crime_data
from train_tfidf import TFIDF
from utils.post_cleaning import process_text

from sklearn.model_selection import train_test_split
from sklearn import metrics
from gensim.models import TfidfModel, Doc2Vec
from gensim.corpora import Dictionary
import pickle
import numpy as np


class CrimeTypeTest():
    def __init__(self, hybridmode=0):
        self.load = False
        # self.tfidf = TFIDF.load("data/models/tfidf.modelFile")
        self.doc2vec = Doc2Vec.load("data/models/word2vec.modelFile")
        self.rulebased = RuleBasedClassifier()
        self.hybrid = HybridClassifier(hybridmode)
        self.number = 0
        self.__split_dataset()

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
            if self.number > 0:
                unlabelled_e = collect_unlabelled_data(170, self.number)
                unlabelled_s = collect_unlabelled_data(92, self.number)
                unlabelled_r = collect_unlabelled_data(
                    114, 0, nocrypt=True)
                unlabelled_c = collect_unlabelled_data(299, 0)
                self.U_train = unlabelled_e + unlabelled_s + unlabelled_r + unlabelled_c
                with open("cluster.data", "wb") as f:
                    pickle.dump(self.U_train, f)
            else:
                unlabelled_e = []
                unlabelled_s = []
                unlabelled_r = []
                unlabelled_c = []
                self.U_train = unlabelled_e + unlabelled_s + unlabelled_r + unlabelled_c
        print("Collected")

        rem, X_et = train_test_split(
            labelled_e, test_size=0.1, shuffle=False)
        X_e, X_ev = train_test_split(
            rem, test_size=0.11, shuffle=True)
        rem, X_st = train_test_split(
            labelled_s, test_size=0.1, shuffle=False)
        X_s, X_sv = train_test_split(
            rem, test_size=0.11, shuffle=True)
        rem, X_rt = train_test_split(
            labelled_r, test_size=0.1, shuffle=False)
        X_r, X_rv = train_test_split(
            rem, test_size=0.11, shuffle=True)
        rem, X_ct = train_test_split(
            labelled_c, test_size=0.1, shuffle=False)
        X_c, X_cv = train_test_split(
            rem, test_size=0.11, shuffle=True)

        self.X_train = X_e + X_s + X_r + X_c
        self.X_val = X_ev + X_sv + X_rv + X_cv
        self.X_test = X_et + X_st + X_rt + X_ct

    def validate_kmeans(self, embedding="d2v", number=500):
        # if number != self.number:
        #     self.number = number
        for i in range(0, 1):
            self.__split_dataset()
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
            self.hybrid.train({'kmeans': self.kmeans}, None)
            print(
                "======================================================================== " + str(i))
            self.kmeans.test(self.X_val)
            self.hybrid.mode = 0
            self.hybrid.test(self.X_val)
            self.hybrid.mode = 1
            self.hybrid.test(self.X_val)

    def test_kmeans(self):
        self.kmeans.test(self.X_test)

    def test_rb(self):
        self.rulebased.test(self.X_test)

    def test_hybrid(self):
        self.hybrid.test(self.X_test)

    def test_deployment(self, data):
        self.kmeans.test(data)


ctt = CrimeTypeTest()
ctt.validate_kmeans(number=0)
ctt.test_kmeans()
ctt.test_rb()
ctt.test_hybrid()
# with open("data/crime.data", "rb") as f:
#     data = pickle.load(f)

# new_data = []
# for d in data:
#     new_data.append((process_text(d[0]), int(d[1])))


# ctt.test_deployment(new_data)
# with open("data/noncrime.data", "rb") as f:
#     results = pickle.load(f)

# g = ctt.kmeans.groundtruths
# p = ctt.kmeans.predictions
# for r in results[:110]:

#     g.append(-1 if r[0] == 0 else r[0])
#     p.append(-1 if r[1] == 0 else r[1])

# print(metrics.classification_report(g, p, digits=3))
