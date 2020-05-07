import abc

from sklearn import metrics
import csv


class Classifier(abc.ABC):

    def __init__(self, embedding_model=None):
        super().__init__()
        self.embedding_model = embedding_model
        self.filename = ""
        self.params = None

    @abc.abstractmethod
    def train(self, training_data: dict, params: dict):
        pass

    @abc.abstractmethod
    def pred(self, example: list):
        pass

    def test(self, test_data: list):
        self.groundtruths = []
        self.predictions = []
        for td in test_data:
            self.groundtruths.append(td[1])
            self.predictions.append(self.pred(td[0]))

    def present_results(self):
        accuracy = metrics.accuracy_score(self.groundtruths, self.predictions)
        precision = metrics.precision_score(
            self.groundtruths, self.predictions)
        recall = metrics.recall_score(self.groundtruths, self.predictions)
        f1 = metrics.f1_score(self.groundtruths, self.predictions)

        to_write = [str(p) for p in self.params.values()]

        to_write.extend([str(accuracy),
                         str(precision),
                         str(recall),
                         str(f1)])

        with open(self.filename, "a") as f:
            writer = csv.writer(
                f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(to_write)

        print(metrics.classification_report(
            self.groundtruths, self.predictions, digits=3))
