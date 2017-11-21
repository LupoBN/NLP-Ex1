import DataManager
import sys
import random
import numpy as np
from Helpers import test_model


class DummyProbsProvider():
    def __init__(self):
        pass

    def get_q(self, y, t1, t2):
        """simulates the conditional probability q(y|t1t2)"""
        return random.random()

    def get_e(self, x, y):
        """simulates the conditional probability of a word x given a pos y"""
        return random.random()


class GreedyTagger:
    def __init__(self, probs_provider):
        self.probs = probs_provider

    def predict_tags(self, words):
        """

        :param words: an ordered list of all words in the sentence to predict. starts with a start symbol
        :return: predictions: an ordered list of the predicted labels
        """
        predictions = ["Start", "Start"]
        words = ["^^^^"] + words + ["", ""]
        for i in range(2, len(words) - 2):  # ignore the first start tag

            labels_probs = self.probs.get_probabilities(
                [words[i], words[i - 1], words[i - 2], words[i + 1], words[i + 2]], predictions[-1], predictions[-2])
            best_label = max(labels_probs, key=labels_probs.get)
            predictions.append(best_label)
        return predictions[2:]  # ignore the two start tags


if __name__ == '__main__':
    probability_provider = DataManager.ProbabilityContainer(sys.argv[3], sys.argv[2])
    gt = GreedyTagger(probability_provider)

    print test_model(sys.argv[1], gt)
