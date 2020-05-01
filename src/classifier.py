import abc


class Classifier(abc.ABC):

    def __init__(self, embedding_model=None):
        super().__init__()
        self.embedding_model = embedding_model

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
