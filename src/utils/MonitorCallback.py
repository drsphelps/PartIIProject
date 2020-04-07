import logging
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile


class MonitorCallback(CallbackAny2Vec):

    def __init__(self, path):
        self.epoch = 0
        self.path = path

    def on_epoch_end(self, model):
        logging.info("Epoch " + str(self.epoch) + " completed")
        output_path = get_tmpfile(
            '{}_epoch{}.model'.format(self.path, self.epoch))
        model.save(output_path)
        self.epoch += 1
