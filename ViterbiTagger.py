import DataManager
import numpy as np
import sys
from Helpers import test_model


class ViterbiTagger:
    def __init__(self, probs_provider, possible_labels):
        self.labels_set = list(probs_provider.get_label_set())
        self.possible_labels = possible_labels
        self.probs = probs_provider

    def predict_tags(self, words):
        """

        :param words: an ordered list of all words in the sentence to predict. starts with a start symbol
        :return: predictions: an ordered list of the predicted labels
        """

        """initialization """
        V, bp = [], []
        words = words + ["", ""]
        n = len(words) - 2
        l = len(self.labels_set)
        # for i in range(n):
        #     V.append([[0. for j in range(len(self.labels_set))] for k in range(len(self.labels_set))])
        #     bp.append([[0. for j in range(len(self.labels_set))] for k in range(len(self.labels_set))])
        #
        V = np.zeros((n, l, l))
        bp = np.zeros((n, l, l), dtype=int)

        start_key = self.labels_set.index("Start")

        V[0][start_key][start_key] = 1.
        V[1][start_key][start_key] = 1.
        bp[0][start_key][start_key] = start_key
        bp[1][start_key][start_key] = start_key

        V = np.where(V > 1e-9, np.log(V), -np.inf)

        """ compute the prob. of a sequence of length i that ends with the labels t, r"""

        for i in range(1, n):  # skip the start symbol.
            word = words[i]

            word_possible_labels = list(self.possible_labels[word]) if word in self.possible_labels else self.labels_set
            prev_word = words[i - 1]
            prev_possible_labels = list(
                self.possible_labels[prev_word]) if prev_word in self.possible_labels else self.labels_set
            prev_prev_word = words[i - 2] if i >= 2 else "^^^^^"
            prev_prev_possible_labels = list(
                self.possible_labels[prev_prev_word]) if prev_prev_word in self.possible_labels else self.labels_set
            next_word = words[i + 1]
            next_next_word = words[i + 2]
            for t in prev_possible_labels:
                t_index = self.labels_set.index(t)

                for r in word_possible_labels:
                    r_index = self.labels_set.index(r)
                    max_val, max_t_prime = -np.inf, None

                    for t_prime in prev_prev_possible_labels:

                        t_prime_index = self.labels_set.index(t_prime)
                        V_prev_t_t_prime = V[i - 1][t_prime_index][t_index]

                        score = self.probs.get_score([word, prev_word, prev_prev_word, next_word, next_next_word], r,
                                                     t, t_prime) + V_prev_t_t_prime

                        if score > max_val:
                            max_val = score
                            max_t_prime = t_prime_index

                    V[i][t_index][r_index] = max_val
                    bp[i][t_index][r_index] = max_t_prime
        """
        calcualte max on t, r of V[n-1]
        """
        V_last = V[-1]
        y_n_minus1, y_n = np.unravel_index(np.argmax(V_last), V_last.shape)

        y = [0] * n
        y[-2], y[-1] = y_n_minus1, y_n
        for i in range(n - 3, 0, -1):
            y[i] = bp[i + 2][y[i + 1]][y[i + 2]]

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
    words_and_labels = DataManager.read_file("data/ass1-tagger-train", DataManager.parse_pos_reading)
    possible_labels = DataManager.parse_possible_labels(words_and_labels)

    probability_provider = DataManager.ProbabilityContainer(sys.argv[3], sys.argv[2])

    vt = ViterbiTagger(probability_provider, possible_labels)
    print test_model(sys.argv[1], vt)
