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

class ViterbiTagger:
    def __init__(self, labels_set):
        self.labels_set = list(labels_set)
        self.probs = DummyProbsProvider()

    def predict_tags(self, words):
        """

        :param words: an ordered list of all words in the sentence to predict. starts with a start symbol
        :return: predictions: an ordered list of the predicted labels
        """

        """initialization """
        V, bp = [], []
        n = len(words)
        for i in range(n):
            V.append([[0. for j in range(len(self.labels_set))] for k in range(len(self.labels_set))])
            bp.append([[0. for j in range(len(self.labels_set))] for k in range(len(self.labels_set))])


        start_key = self.lables_set.index("START")
        V[0][start_key][start_key] = 1.
        bp[0][start_key][start_key] = "START"

        """ compute the prob. of a sequence of length i that ends with the labels tr"""

        for i in range(1, n): #skip two first start symbols.

            for t_index, t in enumerate(self.labels_set):
                for r_index, r in enumerate(self.labels_set):
                   max_val, max_t_prime = -1, None
                   for t_prime_index, t_prime in enumerate(self.labels_set):
                       V_prev_t_t_prime = V[i-1][t_prime_index][t_index]
                       q = self.probs.get_q(r, t_prime, t)
                       r = self.probs.get_e(words[i], r)
                       if V_prev_t_t_prime * q * r > max_val:
                           max_val = V_prev_t_t_prime * q * r
                           max_t_prime = t_prime_index

                   V[i][t_index][r_index] = max_val
                   bp[i][t_index][r_index] = max_t_prime
        """
        calcualte max on t, r of V[n-1]
        """
        V_last = V[n-1]
        max_val = -1
        y_n, y_n_minus1 = None, None


        for i, t in enumerate(self.labels_set):
            for j, r in enumerate(self.labels_set):
                if V_last[i][j] > max_val:
                    max_val = V_last[i][j]
                    y_n, y_n_minus1 = i, j

        y = [None] * n-1
        y[-1], y[-2] = y_n, y_n_minus1
        for i in range(n-3, 0, -1):
            y[i] = bp[i+2][y[i+1]][y[i+2]]

        predictions  = []
        for i, l in enumerate(y):
            predictions[i] = self.labels_set.index(l)

        return predictions