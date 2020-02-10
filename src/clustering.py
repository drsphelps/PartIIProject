import random
import numpy as np
from gensim.models import Doc2Vec
from train_embeddings import MonitorCallback
from post_cleaning import process_text
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle


d2v_model = Doc2Vec.load('data/models/1.modelFile')


def nthmin(a, n):
    return np.partition(a, n)[n-1]


class KMeansPoint():
    def __init__(self, coords, constraint=-1):
        self.coords = coords
        self.constraint = constraint
        self.cluster = -1

    def setCluster(self, cluster):
        self.cluster = cluster


class ConstrainedKMeans():
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500, max_violations=10):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.max_violations = max_violations
        self.num_constraints = 0

    def violateConstraints(self, cluster, point):
        violated = 0
        for c in cluster:
            if point.constraint != c.constraint:
                violated += 1
        return violated

    def train(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i].coords

        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            # find the distance between the point and cluster; choose the nearest centroid
            for d in data:
                distances = [np.linalg.norm(
                    d.coords - self.centroids[centroid]) for centroid in self.centroids]

                n = 0
                classification = distances.index(nthmin(distances, n))
                while self.violateConstraints(self.classes[classification], d) > self.max_violations:
                    n += 1
                    classification = distances.index(nthmin(distances, n))

                self.classes[classification].append(d)

            previous = dict(self.centroids)
            for classification in self.classes:
                a = [x.coords for x in self.classes[classification]]
                self.centroids[classification] = np.average(a, axis=0)

            isOptimal = True

            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            if isOptimal:
                break

    def pred(point):
        distances = [np.linalg.norm(
            d.coords - self.centroids[centroid]) for centroid in self.centroids]
        return distances.index(min(distances))


def load_crime_data(folder, c):
    tests = []
    for i in range(0, 500):
        with open('data/' + folder + '_data/' + str(i) + '.data', 'r') as f:
            tests.append(d2v_model.infer_vector(process_text(f.read())))

    tests = [KMeansPoint(x, c) for x in tests]
    return tests


def main():
    done = False
    max_min = 0

    while not done:
        X = load_crime_data('ewhore', 0) + load_crime_data("stresser", 1)

        X = np.stack(X)

        km = ConstrainedKMeans(2)
        km.train(X)

        if min([len(i) for i in km.classes.values()]) > max_min:
            colors = 10*["r", "g", "c", "b", "k"]
            markers = ['.', 'x']

            for centroid in km.centroids:
                plt.scatter(km.centroids[centroid][0],
                            km.centroids[centroid][1], s=130, marker="x")

            for classification in km.classes:
                color = colors[classification]
                embedded = [x.coords for x in km.classes[classification]]
                embedded = TSNE(2).fit_transform(embedded)
                for i in range(0, len(embedded)):
                    plt.scatter(embedded[i][0], embedded[i][1],
                                color=colors[km.classes[classification][i].constraint], s=30, marker=markers[classification])

            with open("bestpoints.b", "wb") as f:
                pickle.dump(km, f)
            max_min = min([len(i) for i in km.classes.values()])
            print(max_min)

        if max_min > 400:
            done = True

        plt.savefig('a.png')


if __name__ == "__main__":
    # km = None
    # with open("bestpoints.b", "rb") as f:
    #     km = pickle.load(f)

    # colors = 10*["r", "g", "c", "b", "k"]
    # markers = ['.', 'x']

    # # for centroid in km.centroids:
    # #     plt.scatter(km.centroids[centroid][0],
    # #                 km.centroids[centroid][1], s=130, marker="x")

    # for classification in km.classes:
    #     color = colors[classification]
    #     embedded = [x.coords for x in km.classes[classification]]
    #     embedded = TSNE(2, 100).fit_transform(embedded)
    #     for i in range(0, len(embedded)):
    #         plt.scatter(embedded[i][0], embedded[i][1],
    #                     color=colors[km.classes[classification][i].constraint], s=30, marker=markers[classification])

    # plt.savefig("test.png")

    main()
