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


class KMeansPoint():
    def __init__(self, coords, constraint=-1):
        self.coords = coords
        self.constraint = constraint
        self.cluster = -1

    def setCluster(self, cluster):
        self.cluster = cluster


class ConstrainedKMeans():
    LOAD = True

    def __init__(self, k=3, metric=(lambda x, y: np.linalg.norm(x-y)),
                 tolerance=0.0001, max_iterations=500, max_violations=100):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.max_violations = max_violations
        self.num_constraints = 0
        self.metric = metric
        self.labels = [0] * k

    def violateConstraints(self, cluster, point):
        # print("\n\n")
        # print("Cluster size: " + str(len(cluster)))
        violated = 0
        # print("Point Constraint " + str(point.constraint))
        print_constraints(cluster)
        if point.constraint != -1:
            for c in cluster:
                if c.constraint != -1:
                    if point.constraint != c.constraint:
                        # print("Broken Constraint " +
                        #       str(point.constraint) + " " + str(c.constraint))
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
        print(self.labels)

    def train(self, l_data, u_data):

        self.centroids = {}

        self.classes = {}
        for i in range(self.k):
            self.classes[i] = [d for d in l_data if d.constraint == i]

        for i in range(self.k):
            self.centroids[i] = np.sum(
                np.array([d.coords for d in self.classes[i]]), axis=0)/len(self.classes[i])

        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = [d for d in l_data if d.constraint == i]

            # find the distance between the point and cluster; choose the nearest centroid
            for d in u_data:
                distances = [self.metric(d.coords, self.centroids[centroid])
                             for centroid in self.centroids]
                # print("Distances to centres: " + str(distances))

                n = 0
                classification_order = np.argsort(distances)
                # print("Adding to cluster: " + str(classification_order[n]))
                # while self.violateConstraints(self.classes[classification_order[n]], d):
                # print(str(n) + ": " + str(d.constraint) +
                #       " " + str(classification_order[n]))
                # n += 1
                # if n >= len(self.centroids):
                #     return False
                #     classification = distances.index(nthmin(distances, 0))
                #     break
                # print("Adding to cluster: " + str(classification_order[n]))

                self.classes[classification_order[n]].append(d)
                # print(lengths(self.classes.values()))
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

    def pred(self, point):
        distances = [self.metric(point.coords, self.centroids[centroid])
                     for centroid in self.centroids]
        return self.labels[distances.index(min(distances))] == point.constraint

    @staticmethod
    def collect_training_data(split, number_unlabelled):
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
            for post in conn.get_posts_from_forum(forum, number_unlabelled):
                posts.append(post[0])
            conn.close_connection()
            print("collected unlabelled")
            return posts

        # Get labelled data
        labelled_e = load_labelled_data("ewhore")
        labelled_s = load_labelled_data("stresser")
        labelled_r = load_labelled_data("rat")
        labelled_c = load_labelled_data("crypter")

        # Collect or load unlabelled data
        if ConstrainedKMeans.LOAD:
            with open("cluster.data", "rb") as f:
                U_train = pickle.load(f)
            print("collected unlabelled")
        else:
            unlabelled_e = collect_unlabelled_data(170)
            unlabelled_s = collect_unlabelled_data(92)
            unlabelled_r = collect_unlabelled_data(114)
            unlabelled_c = collect_unlabelled_data(299)
            U_train = unlabelled_e + unlabelled_s + unlabelled_r + unlabelled_c
            with open("cluster.data", "wb") as f:
                pickle.dump(U_train, f)
                ConstrainedKMeans.LOAD = True

        X_e, X_et = train_test_split(labelled_e, test_size=split, shuffle=True)
        X_s, X_st = train_test_split(labelled_s, test_size=split, shuffle=True)
        X_r, X_rt = train_test_split(labelled_r, test_size=split, shuffle=True)
        X_c, X_ct = train_test_split(labelled_c, test_size=split, shuffle=True)

        X_train = X_e + X_s + X_r + X_c
        X_test = X_et + X_st + X_rt + X_ct

        train_size = int((1.-split) * 500)
        test_size = int(split * 500)

        constraints = ([0] * train_size) + ([1] * train_size) + \
            ([2] * train_size) + ([3] * train_size)

        test_constraints = [0] * test_size + [1] * \
            test_size + [2] * test_size + [3] * test_size

        for i in range(len(X_train)):
            X_train[i] = KMeansPoint(d2v_model.infer_vector(
                process_text(X_train[i])), constraint=constraints[i])
        X_train = np.stack(X_train)

        for i in range(len(U_train)):
            U_train[i] = KMeansPoint(d2v_model.infer_vector(
                process_text(U_train[i])), -1)
        U_train = np.stack(U_train)

        for i in range(len(X_test)):
            X_test[i] = KMeansPoint(d2v_model.infer_vector(
                process_text(X_test[i])), constraint=test_constraints[i])
        X_test = np.stack(X_test)

        return X_train, U_train, X_test

    @staticmethod
    def test():
        X_train, U_train, X_test = ConstrainedKMeans.collect_training_data(
            0.1, 1000)
        km = ConstrainedKMeans(
            4, lambda x, y: py_ang(x, y), max_violations=500)

        if km.train(X_train, U_train):
            print("Trained")
            print("Accuracy")
            correct = 0.
            for test in range(X_test.shape[0]):
                if km.pred(X_test[test]):
                    correct += 1.
            print(correct / X_test.shape[0])
            return correct / X_test.shape[0]


def main():
    done = False
    max_min = 0

    for _ in range(10):
        while not done:
            with open("xtrain.data", "rb") as f:
                X_train = pickle.load(f)
            with open("xtest.data", "rb") as f:
                X_test = pickle.load(f)

            u_train = np.array([d for d in X_train if d.constraint == -1])

            km = ConstrainedKMeans(
                4, lambda x, y: py_ang(x, y), max_violations=500)
            print("Training")
            if km.train(X_train, u_train):
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
                break

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

    n = 10.
    accuracy = 0.
    for i in range(int(n)):
        accuracy += ConstrainedKMeans.test()
    print("Total: " + str(accuracy/n))
