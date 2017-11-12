#import DataManager
import random

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
    def __init__(self, labels_set):
        self.labels_set = labels_set
        self.probs = DummyProbsProvider()

    def predict_tags(self, words):
        """

        :param words: an ordered list of all words in the sentence to predict. tarts with a start symbol
        :return: predictions: an ordered list of the predicted labels
        """

        predictions = ["Start", "Start"]
        for word in words[1:]: # ignore the first start tag
            best_label = None
            best_prob = -1
            for i, current_label in enumerate(self.labels_set):
                last_two = predictions[-2], predictions[-1]
                e, q = self.probs.get_e(word, current_label),\
                       self.probs.get_q(word, last_two[0], last_two[1])
                if e * q > best_prob:
                    best_label, best_prob = current_label, e*q
            predictions.append(best_label)

        return predictions[2:] #ignore the two start tags

labels = {"NN", "NP", "V"}
words = "hello how are you".split(" ")
gt = GreedyTagger(labels)
print gt.predict_tags(words)
