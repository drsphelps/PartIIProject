from classifier import Classifier
from utils.kmeans_point import KMeansPoint

import numpy as np
import pickle
from sklearn import metrics


class KMeansClassifier(Classifier):
    def __init__(self, embedding_model):
        super().__init__(embedding_model)
        self.labels = []

    def __format_data(self, data):
        formatted = []
        for d in data:
            embedding = self.embedding_model.infer_vector(d[0])
            formatted.append(KMeansPoint(embedding, d[1]))
        return formatted

    def calculate_label(self, classes, k):
        for i in range(len(classes)):
            c = [0] * k
            for point in classes[i]:
                if point.constraint != -1:
                    c[point.constraint] += 1
            self.labels.append(c.index(max(c)))

    def train(self, training_data: dict, params: dict):
        X_train = self.__format_data(training_data['X_train'])
        U_train = self.__format_data(training_data['U_train'])
        k = params['k']
        max_iterations = params['max_iterations']
        self.metric = params['metric']
        tolerance = params['tolerance']

        self.centroids = {}

        classes = {}
        for i in range(k):
            classes[i] = [d for d in X_train if d.constraint == i]

        for i in range(k):
            self.centroids[i] = np.sum(
                np.array([d.coords for d in classes[i]]), axis=0)/len(classes[i])

        for i in range(max_iterations):
            classes = {}
            for i in range(k):
                classes[i] = [
                    d for d in X_train if d.constraint == i]

            # find the distance between the point and cluster; choose the nearest centroid
            for d in U_train:
                distances = [self.metric(d.coords, self.centroids[centroid])
                             for centroid in self.centroids]

                classification_order = np.argsort(distances)
                classes[classification_order[0]].append(d)

            previous = dict(self.centroids)
            for classification in classes:
                coords = [x.coords for x in classes[classification]]
                self.centroids[classification] = np.average(coords, axis=0)

            optimal = True

            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid)/original_centroid * 100.0) > tolerance:
                    optimal = False
            if optimal:
                self.calculate_label(classes, k)
                break

    def pred(self, example):
        embedding = self.embedding_model.infer_vector(example)
        distances = [self.metric(embedding, self.centroids[centroid])
                     for centroid in self.centroids]
        return self.labels[distances.index(min(distances))]

    def pred_full(self, example):
        embedding = self.embedding_model.infer_vector(example)
        distances = [self.metric(embedding, self.centroids[centroid])
                     for centroid in self.centroids]
        return distances

    def test(self, data):
        super().test(data)
        self.present_results()

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save(obj, filename):
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
