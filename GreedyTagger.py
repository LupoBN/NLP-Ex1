import DataManager
import sys
import random
import numpy as np
import time

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
        for word in words[1:]: # ignore the first start tag
            best_label = None
            best_prob = -1000000

            for i, current_label in enumerate(labels):
                #print "checking label ", current_label
                last_two = predictions[-2], predictions[-1]
                e, q = self.probs.get_e_prob(word, current_label),\
                       self.probs.get_q_prob(current_label, last_two[-1], last_two[-2])
                score = np.log(e) + np.log(q)
                if score > best_prob:
                    best_label, best_prob = current_label, score
            predictions.append(best_label)

        return predictions[2:] #ignore the two start tags




if __name__ == '__main__':
    probability_provider = DataManager.ProbabilityContainer("e.mle", "q.mle" )
    words_orig = "^^^^^ One/NN might/MD think/VB that/IN the/DT home/NN fans/NNS in/IN this/DT Series/NNP of/IN the/DT Subway/NNP Called/VBN BART/NNP (/( that/DT 's/VBZ a/DT better/JJR name/NN for/IN a/DT public/JJ conveyance/NN than/IN ``/`` Desire/NN ,/, ''/'' do/VBP n't/RB you/PRP think/VBP ?/. )/) would/MD have/VB been/VBN ecstatic/JJ over/IN the/DT proceedings/NNS ,/, but/CC they/PRP observe/VBP them/PRP in/IN relative/JJ calm/NN ./.Partisans/NNS of/IN the/DT two/CD combatants/NNS sat/VBD side/NN by/IN side/NN".split(" ")
    words = [word.split("/")[0] for word in words_orig]

    gt = GreedyTagger( probability_provider)
    f = open("data/ass1-tagger-test")
    lines = f.readlines()

    good, bad = 0., 0.

    for i in range(len(lines)):

        words_orig = ("^^^^^/Start " + lines[i]).split(" ")
        words = [word.split("/")[0] for word in words_orig]
        labels = [word.split("/")[1] for word in words_orig]

        start = time.time()
        preds = gt.predict_tags(words)
        print time.time() - start

        s = ""
        print labels[1:]
        for i, word in enumerate(words[1:]):
            s += word + "(" + preds[i] + ") "
            if preds[i] == labels[i + 1]:
                good += 1
            else:
                bad += 1
        print s
        print words_orig[1:]
        print "accuracy: ", (good) / (good + bad)