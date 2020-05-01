from clustering import ConstrainedKMeans
from sklearn import metrics
from get_training import collect_training_data
from utils.MonitorCallback import MonitorCallback
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from train_tfidf import infer_tfidf
import pickle


class HybridClassifier():
    def __init__(self):
        self.words = [["ewhor", "e-whor"],
                      ["stresser", "booter"], [" rat "], ["crypt", "fud"]]
        # with open("data/models/word2vec.modelFile", "rb") as f:
        #     self.d2v_model = pickle.load(f)
        self.tfidf_model = TfidfModel.load("data/models/1tfidf.modelFile")
        self.dct = Dictionary.load("data/models/1tfidf.dct")

    def __rule_based(self, post):
        max_mentions = 0
        max_class = -1
        for classification in range(len(self.words)):
            mentions = 0
            for word in self.words[classification]:
                if word in post:
                    mentions += 1
            if mentions > max_mentions:
                max_mentions = mentions
                max_class = classification
            elif mentions == max_mentions:
                max_class = -1
        return max_class

    def hybrid_pred(self, kmp, kmeans_model):
        # rb_class = self.__rule_based(kmp.original_text)
        # if rb_class != -1 and rb_class == kmp.constraint:
        #     return rb_class
        # else:
        return kmeans_model.pred_class(kmp)

    def test_hybrid_model(self):
        X_train, U_train, X_test = collect_training_data(
            True, 0.1, 500, infer_tfidf(self.tfidf_model, self.dct))
        kmeans_model = ConstrainedKMeans(4)
        if kmeans_model.train(X_train, U_train):
            correct = 0.
            classifications = []
            groundtruths = []
            for test in range(X_test.shape[0]):
                classification = self.hybrid_pred(X_test[test], kmeans_model)
                classifications.append(classification)
                groundtruths.append(X_test[test].constraint)
                if classification == X_test[test].constraint:
                    correct += 1.
        print(metrics.classification_report(
            groundtruths, classifications, digits=3))
        return correct / X_test.shape[0]


h = HybridClassifier()
h.test_hybrid_model()
