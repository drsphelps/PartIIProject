import random
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors


class Sampling():
    """
    Used to over/undersample a dataset. The dataset labels must be one-hot vectors, 
    and there must be a majority class for undersampling to work. 
    """

    def __init__(self, oversample_rate=1., undersample_rate=1.):
        self.k = 5
        self.osr = oversample_rate
        self.usr = undersample_rate

    def __find_maj_class(self, labels):
        return stats.mode(labels)[0][0].tolist()

    def __undersample(self, data):
        data = np.stack(data)
        length = data.shape[0]
        return data[np.random.choice(length, int(length * self.usr), replace=False)]

    def __resample(self, data):
        data = np.stack(data)
        length = data.shape[0]
        return data[np.random.choice(length, int(length * self.osr), replace=True)]

    def __populate(self, num_syn, sample, nnarray):
        syn = []
        while num_syn > 0:
            new = []
            k = random.randint(0, self.k - 1)
            for attr in range(0, sample.shape[0]):
                dif = nnarray[k][attr] - sample[attr]
                gap = random.random()
                new.append(sample[attr] + (dif*gap))
            syn.append(new)
            num_syn -= 1
        return syn

    def __smote(self, data):
        nbrs = NearestNeighbors(n_neighbors=self.k+1,
                                algorithm='ball_tree').fit(data)
        distance, indices = nbrs.kneighbors(data)
        num_syn = int(self.osr) - 1

        synthetic = []

        for i in range(0, len(data)):
            nnarray = []
            for k in range(1, self.k+1):
                nnarray.append(data[indices[i][k]])
            new_points = self.__populate(num_syn, data[i], nnarray)
            synthetic.extend(new_points)
        if len(synthetic) == 0:
            new_min = np.stack(data)
        else:
            synthetic = np.stack(synthetic)
            new_min = np.concatenate((data, synthetic), axis=0)
        return new_min

    def perform_sampling(self, data, labels, min_label=None):
        maj_label = self.__find_maj_class(labels)
        maj_data = []
        min_data = []
        other_data = []
        other_labels = []
        for i in range(0, labels.shape[0]):
            l = labels[i].tolist()
            if l == maj_label:
                maj_data.append(data[i])
            else:
                if min_label is not None and l == min_label:
                    min_data.append(data[i])
                else:
                    other_data.append(data[i])
                    other_labels.append(l)

        maj_data = self.__undersample(maj_data)
        min_data = self.__smote(min_data)
        data = np.concatenate((maj_data, min_data), axis=0)
        labels = np.concatenate((np.array(
            [maj_label] * maj_data.shape[0]), np.array([min_label] * min_data.shape[0])))
        return data, labels
