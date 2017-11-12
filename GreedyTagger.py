import DataManager
import sys
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
    def __init__(self, labels_set, probs_provider):
        self.labels_set = labels_set
        self.probs = probs_provider

    def predict_tags(self, words):
        """

        :param words: an ordered list of all words in the sentence to predict. starts with a start symbol
        :return: predictions: an ordered list of the predicted labels
        """

        labels = self.probs.get_label_set()
        predictions = ["Start", "Start"]
        for word in words[1:]: # ignore the first start tag
            best_label = None
            best_prob = -1

            for i, current_label in enumerate(labels):
                #print "checking label ", current_label
                last_two = predictions[-2], predictions[-1]
                e, q = self.probs.get_e_prob(word, current_label),\
                       self.probs.get_q_prob(current_label, last_two[-1], last_two[-2])
                print "e is ", e, " and q is ", q, " word is ", word, " and labels are ", last_two[0], " ", last_two[1], " ", current_label
                assert e > 0
                assert q > 0
                if e * q > best_prob:
                    best_label, best_prob = current_label, e*q
            print "best prob for word ", word, " is ", best_prob
            predictions.append(best_label)

        return predictions[2:] #ignore the two start tags




if __name__ == '__main__':
    probability_provider = DataManager.ProbabilityContainer("e.mle", "q.mle" )
    labels = {"NN", "NP", "V"}
    words = "^^^^^ How are you .".split(" ")
    gt = GreedyTagger(labels, probability_provider)
    print gt.predict_tags(words)