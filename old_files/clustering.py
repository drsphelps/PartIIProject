import numpy as np
from utils.db import db
from gensim.models import Doc2Vec
from utils.MonitorCallback import MonitorCallback
from utils.post_cleaning import process_text
from utils.get_training import collect_training_data
from utils.kmeans_point import KMeansPoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import pickle


def lengths(arr):
    l = []
    for a in arr:
        l.append(len(a))
    return l


def print_constraints(arr):
    cons = []
    for a in arr:
        cons.append(a.constraint)
    # print("List of constraints " + str(set(cons)))


def py_ang(v1, v2):
    dot = np.dot(v1, v2)
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    c = dot/(v1_mag*v2_mag)
    if abs(c) > 1.:
        return 0.0
    return abs(np.arccos(dot/(v1_mag*v2_mag)))


class ConstrainedKMeans():
    LOAD = False

    def __init__(self, k=3, metric=(lambda x, y: np.linalg.norm(x-y)),
                 tolerance=0.0001, max_iterations=500, max_violations=100, nu=1000, split=0.1):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.max_violations = max_violations
        self.number_unlabelled = nu
        self.num_constraints = 0
        self.metric = metric
        self.split = split
        self.labels = [0] * k
        self.d2v_model = Doc2Vec.load('data/models/2.modelFile')

    def violateConstraints(self, cluster, point):
        violated = 0
        print_constraints(cluster)
        if point.constraint != -1:
            for c in cluster:
                if c.constraint != -1:
                    if point.constraint != c.constraint:
                        violated += 1
        return violated > self.max_violations

    def inc(self):
        self.max_violations += 10

    def calculate_label(self):
        for i in range(len(self.classes)):
            c = [0] * self.k
            for point in self.classes[i]:
                if point.constraint != -1:
                    c[point.constraint] += 1
            self.labels[i] = c.index(max(c))

    def train(self, X_train, U_train):

        self.centroids = {}

        self.classes = {}
        for i in range(self.k):
            self.classes[i] = [d for d in X_train if d.constraint == i]

        for i in range(self.k):
            self.centroids[i] = np.sum(
                np.array([d.coords for d in self.classes[i]]), axis=0)/len(self.classes[i])

        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = [
                    d for d in X_train if d.constraint == i]

            # find the distance between the point and cluster; choose the nearest centroid
            for d in U_train:
                distances = [self.metric(d.coords, self.centroids[centroid])
                             for centroid in self.centroids]

                n = 0
                classification_order = np.argsort(distances)
                self.classes[classification_order[n]].append(d)
                for c in self.classes.values():
                    print_constraints(c)

            previous = dict(self.centroids)
            for classification in self.classes:
                if len(self.classes[classification]) != 0:
                    a = [x.coords for x in self.classes[classification]]
                    self.centroids[classification] = np.average(a, axis=0)
                else:
                    self.centroids[classification] = np.array(
                        [100000000000000.] * 100)

            isOptimal = True

            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            if isOptimal:
                self.calculate_label()
                break
        return True

    def test(self, X_train, U_train, X_test):
        if self.train(X_train, U_train):
            correct = 0.
            for test in range(X_test.shape[0]):
                if self.pred(X_test[test]):
                    correct += 1.
            return correct / X_test.shape[0]

    def pred(self, point):
        distances = [self.metric(point.coords, self.centroids[centroid])
                     for centroid in self.centroids]
        return self.labels[distances.index(min(distances))] == point.constraint

    def pred_class(self, point):
        distances = [self.metric(point.coords, self.centroids[centroid])
                     for centroid in self.centroids]
        return self.labels[distances.index(min(distances))]

    def cross_validation(self):
        n = 10.
        accuracy = 0.
        for _ in range(int(n)):
            X_train, U_train, X_test = collect_training_data(
                ConstrainedKMeans.LOAD, self.split, self.number_unlabelled, self.d2v_model.infer_vector)
            ConstrainedKMeans.LOAD = True
            a = self.test(X_train, U_train, X_test)
            accuracy += a
        return accuracy/n

    def d2v_train(self, model):
        self.d2v_model = model
        return self.cross_validation()

    def update(self, k=None, metric=None,
               tolerance=None, max_iterations=None, max_violations=None, nu=None, split=None):
        self.k = self.k if k is None else k
        self.tolerance = self.tolerance if tolerance is None else tolerance
        self.max_iterations = self.max_iterations if max_iterations is None else max_iterations
        self.max_violations = self.max_violations if max_violations is None else max_violations
        self.number_unlabelled = self.number_unlabelled if nu is None else nu
        ConstrainedKMeans.LOAD = False
        self.num_constraints = 0
        self.metric = self.metric if metric is None else metric
        self.split = self.split if split is None else split
        self.labels = [0] * self.k


if __name__ == "__main__":
    kmeans = ConstrainedKMeans(4, nu=0)
    results = {}
    for i in range(500, 4000, 500):
        kmeans.update(nu=i)
        results[i] = kmeans.cross_validation()
        print(results)
