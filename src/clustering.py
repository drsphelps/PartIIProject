import random
import numpy as np
from utils.db import db
from gensim.models import Doc2Vec
from utils.MonitorCallback import MonitorCallback
from post_cleaning import process_text
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import pickle


d2v_model = Doc2Vec.load('data/models/2.modelFile')
LOAD = True


def nthmin(a, n):
    return np.partition(a, n)[n-1]


def py_ang(v1, v2):
    dot = np.dot(v1, v2)
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    c = dot/(v1_mag*v2_mag)
    if abs(c) > 1.:
        return 0.0
    return abs(np.arccos(dot/(v1_mag*v2_mag)))


class KMeansPoint():
    def __init__(self, coords, constraint=-1):
        self.coords = coords
        self.constraint = constraint
        self.cluster = -1

    def setCluster(self, cluster):
        self.cluster = cluster


class ConstrainedKMeans():
    def __init__(self, k=3, metric=(lambda x, y: np.linalg.norm(x-y)),
                 tolerance=0.0001, max_iterations=500, max_violations=2000):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.max_violations = max_violations
        self.num_constraints = 0
        self.metric = metric
        self.labels = [0] * k

    def violateConstraints(self, cluster, point):
        violated = 0
        if point.constraint != -1:
            for c in cluster:
                if point.constraint != c.constraint:
                    violated += 1
        return violated

    def calculate_label(self):
        for i in range(len(self.classes)):
            c = [0] * self.k
            for point in self.classes[i]:
                if point.constraint != -1:
                    c[point.constraint] += 1
            self.labels[i] = c.index(max(c))
        print(self.labels)

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
                distances = [self.metric(d.coords, self.centroids[centroid])
                             for centroid in self.centroids]

                n = 0
                classification = distances.index(nthmin(distances, n))
                while self.violateConstraints(self.classes[classification], d) > self.max_violations:
                    n += 1
                    if n >= len(self.centroids):
                        return False
                    #     classification = distances.index(nthmin(distances, 0))
                    #     break
                    classification = distances.index(nthmin(distances, n))

                self.classes[classification].append(d)

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

    def pred(self, point):
        distances = [self.metric(point.coords, self.centroids[centroid])
                     for centroid in self.centroids]
        return self.labels[distances.index(min(distances))] == point.constraint


def load_labelled_data(folder):
    posts = []
    for i in range(0, 500):
        with open('data/' + folder + '_data/' + str(i) + '.data', 'r') as f:
            posts.append(f.read())
    print("collected labelled")
    return posts


def collect_unlabelled_data(forum):
    conn = db()
    posts = []
    for post in conn.get_posts_from_forum(forum, 1000):
        posts.append(post[0])
    conn.close_connection()
    print("collected unlabelled")
    return posts


def main():
    done = False
    max_min = 0

    # Get labelled data
    labelled_e = load_labelled_data("ewhore")
    labelled_s = load_labelled_data("stresser")
    labelled_r = load_labelled_data("rat")

    # Collect or load unlabelled data
    if LOAD:
        with open("cluster.data", "rb") as f:
            U = pickle.load(f)
        unlabelled_e = U[0:1000]
        unlabelled_s = U[1000:2000]
        print("collected unlabelled")
    else:
        unlabelled_e = collect_unlabelled_data(170)
        unlabelled_s = collect_unlabelled_data(92)
        with open("cluster.data", "wb") as f:
            U = unlabelled_e + unlabelled_s
            pickle.dump(U, f)

    for _ in range(10):
        X_e, X_et = train_test_split(labelled_e, test_size=0.1, shuffle=True)
        X_s, X_st = train_test_split(labelled_s, test_size=0.1, shuffle=True)
        X_r, X_rt = train_test_split(labelled_r, test_size=0.1, shuffle=True)

        X = X_e + X_s + X_r
        X_test = X_et + X_st + X_rt
        constraints = [0] * 450 + [-1] * 0 + [1] * \
            450 + [-1] * 0 + [2] * 450 + [-1] * 0
        test_constraints = [0] * 50 + [1] * 50 + [2] * 50

        Y = X.copy()
        Y_test = X_test.copy()
        while not done:
            X = Y.copy()
            X_test = Y_test.copy()

            for i in range(len(X)):
                X[i] = KMeansPoint(d2v_model.infer_vector(
                    process_text(X[i])), constraint=constraints[i])
            X = np.stack(X)

            for i in range(len(X_test)):
                X_test[i] = KMeansPoint(d2v_model.infer_vector(
                    process_text(X_test[i])), constraint=test_constraints[i])
            X_test = np.stack(X_test)

            km = ConstrainedKMeans(3, lambda x, y: py_ang(x, y))
            print("Training")
            if km.train(X):
                print("Trained")
                if min([len(i) for i in km.classes.values()]) > -1:
                    colors = 10*["r", "g", "c", "b", "k"]
                    markers = ['.', 'x']

                    # for centroid in km.centroids:
                    #     plt.scatter(km.centroids[centroid][0],
                    #                 km.centroids[centroid][1], s=130, marker="x")

                    # for classification in km.classes:
                    #     color = colors[classification]
                    #     embedded = [
                    #         x.coords for x in km.classes[classification]]
                    #     embedded = TSNE(2).fit_transform(embedded)
                    #     for i in range(0, len(embedded)):
                    #         plt.scatter(embedded[i][0], embedded[i][1],
                    #                     color=colors[km.classes[classification][i].constraint], s=30, marker=markers[classification])

                    max_min = min([len(i) for i in km.classes.values()])
                    for c in km.classes.values():
                        print(len(c))

                    print("Accuracy")
                    correct = 0.
                    for test in range(X_test.shape[0]):
                        if km.pred(X_test[test]):
                            correct += 1.
                    print(correct / X_test.shape[0])

                # if max_min > X.shape[0]/2 - 200:
                done = True

                plt.savefig('a.png')
            else:
                print("FAILED")

        done = False


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
