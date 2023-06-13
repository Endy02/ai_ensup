from ai.core.cleaner import Cleaner
from ai.core.analyzer import Analizer
from ai.core.predictor import Predictor

class Quantum():
    def __init__(self):
        self.cleaner = Cleaner()
        self.analizer = Analizer()
        self.predictor = Predictor()

    def make_prediction(self, sample):
        pass

    def prepare_data(self):
        pass

    def __get_pipeline(self):
        pipeline = self.predictor.make_pipeline()
        return pipeline
