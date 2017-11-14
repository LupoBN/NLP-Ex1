import random
import DataManager
import numpy as np
import math
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


class ViterbiTagger:
    def __init__(self, probs_provider, possible_labels):
        self.labels_set = list(probs_provider.get_label_set())
        self.possible_labels  = possible_labels
        self.probs = probs_provider
        #self.labels_set = ["Start", "DT", "NN"]
    def predict_tags(self, words):
        """

        :param words: an ordered list of all words in the sentence to predict. starts with a start symbol
        :return: predictions: an ordered list of the predicted labels
        """

        """initialization """
        V, bp = [], []
        n = len(words)
        l = len(self.labels_set)
        # for i in range(n):
        #     V.append([[0. for j in range(len(self.labels_set))] for k in range(len(self.labels_set))])
        #     bp.append([[0. for j in range(len(self.labels_set))] for k in range(len(self.labels_set))])
        #
        epsilon = 0.
        V = epsilon * np.ones((n, l, l))
        bp = np.zeros((n, l, l), dtype=int)

        start_key = self.labels_set.index("Start")

        V[0][start_key][start_key] = 1.
        V[1][start_key][start_key] = 1.
        bp[0][start_key][start_key] = start_key
        bp[1][start_key][start_key] = start_key

        #V = np.log(V)
        #print V>-1
        """ compute the prob. of a sequence of length i that ends with the labels t, r"""
        s=""
        epsilon = 1e-8
        for i in range(1, n): #skip two first start symbols.
            word = words[i]

            if word in self.possible_labels:
                word_possible_labels = list(self.possible_labels[word])
            else:
                word_possible_labels = self.labels_set

            prev_word = words[i-1]
            prev_possible_labels = list(self.possible_labels[prev_word]) if prev_word in self.possible_labels else self.labels_set
            prev_prev_word = words[i-2] if i>=2 else "^^^^^"
            prev_prev_possible_labels = list(self.possible_labels[prev_prev_word]) if prev_prev_word in self.possible_labels else self.labels_set

            #print "possible labels for word ", word ," are", word_possible_labels
            for t in prev_possible_labels:
                t_index = self.labels_set.index(t)

                for r in word_possible_labels:
                   r_index = self.labels_set.index(r)
                   max_val, max_t_prime = -float("inf"), None

                   for t_prime in prev_prev_possible_labels:
                       t_prime_index = self.labels_set.index(t_prime)
                       V_prev_t_t_prime = V[i-1][t_prime_index][t_index]
                       #print "IT'S ", V_prev_t_t_prime

                       q = self.probs.get_q_prob(r, t, t_prime)
                       e = self.probs.get_e_prob(word, r)

                       #score = np.log(q) + np.log(e) + V_prev_t_t_prime
                       score = q * e * V_prev_t_t_prime
                       if score > max_val:
                           max_val = score
                           max_t_prime = t_prime_index

                   V[i][t_index][r_index] = max_val
                   bp[i][t_index][r_index] = max_t_prime
        """
        calcualte max on t, r of V[n-1]
        """
        V_last = V[-1]
        y_n_minus1, y_n = np.unravel_index(np.argmax(V_last),V_last.shape)

        y = [0]*n
        y[-2], y[-1] = y_n_minus1, y_n
        for i in range(n-3, 0, -1):
            y[i] = bp[i+2][y[i+1]][y[i+2]]

        preds = [self.labels_set[val] for val in y]
        return preds[1:]

"""
labels_set = {"START", "NN", "NP", "V"}
words = "START hello how are you ? I am alright".split(" ")


f =  open("data/possible_labels.txt")
possible_labels = {}
for line in f.readlines():
    parsed_line = line.split('\t')
    word, labels = parsed_line[0], parsed_line[1:]
    possible_labels[word] = labels

gt = ViterbiTagger(labels_set, possible_labels)
print gt.predict_tags(words)
"""

if __name__ == '__main__':
    words_and_labels = DataManager.read_file("data/ass1-tagger-test", DataManager.parse_pos_reading)
    possible_labels = DataManager.parse_possible_labels(words_and_labels)

    probability_provider = DataManager.ProbabilityContainer("e.mle", "q.mle" )
    #words_orig = "^^^^^ One/NN might/MD think/VB that/IN the/DT home/NN fans/NNS in/IN this/DT Series/NNP of/IN the/DT Subway/NNP Called/VBN BART/NNP (/( that/DT 's/VBZ a/DT better/JJR name/NN for/IN a/DT public/JJ conveyance/NN than/IN ``/`` Desire/NN ,/, ''/'' do/VBP n't/RB you/PRP think/VBP ?/. )/) would/MD have/VB been/VBN ecstatic/JJ over/IN the/DT proceedings/NNS ,/, but/CC they/PRP observe/VBP them/PRP in/IN relative/JJ calm/NN ./.Partisans/NNS of/IN the/DT two/CD combatants/NNS sat/VBD side/NN by/IN side/NN".split(" ")
    f = open("data/ass1-tagger-test")
    lines = f.readlines()
    vt = ViterbiTagger(probability_provider, possible_labels)

    good, bad = 0., 0.
    point_error = 0.

    for i in range(len(lines)):

        words_orig = ("^^^^^/Start "+lines[i] ).split(" ")
        print words_orig
        words = [word.split("/")[0] for word in words_orig]
        labels =  [word.split("/")[1].strip("\n") for word in words_orig]

        start = time.time()
        preds =  vt.predict_tags(words)
        print time.time() - start

        s = ""
        for i, word in enumerate(words[1:]):
            s+=word+ "("+preds[i]+") "
            if preds[i] == labels[i+1]:
                good += 1
            else:
                if preds[i]==".":
                    point_error+=1
                print "ERROR: predicted ", preds[i], " for word ", word, " while true label is ", labels[i+1], ". prev word is ", words[i-1]
                bad += 1
        print s
        print words_orig[1:]
        print "accuracy: ", (good)/(good+bad)
