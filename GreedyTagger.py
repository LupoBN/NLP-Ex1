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

        labels = self.probs.get_label_set()
        predictions = ["Start", "Start"]
        for word in words[1:]:  # ignore the first start tag
            best_label = None
            best_prob = -np.inf

            for i, current_label in enumerate(labels):
                # print "checking label ", current_label
                last_two = predictions[-2], predictions[-1]
                score = self.probs.get_score(word, current_label, last_two[-1], last_two[-2])

                if score > best_prob:
                    best_label, best_prob = current_label, score
            predictions.append(best_label)

        return predictions[2:]  # ignore the two start tags


if __name__ == '__main__':
    probability_provider = DataManager.ProbabilityContainer(sys.argv[3], sys.argv[2])
    gt = GreedyTagger(probability_provider)
    print test_model(sys.argv[1], gt)
