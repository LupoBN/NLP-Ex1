import numpy as np
from Helpers import *
SUFFIXES = ["ing", "ed", "s", "er", "est", "dom", "ism", "ist", "al", "ity", "ment", "ness", "tion",
            "ship", "ate", "en", "ify", "fy", "sion", "ize", "able", "ful", "ish", "less", "ive"]
PREFIXES = ["un", "de", "re", "in", "anti", "auto", "Auto", "Anti", "Un", "De", "Re", "In", "im", "Im", "Pre", "pre", "extra", "Extra", "over", "Over"]

def count_labels(labels, order):
    labels_count = dict()
    labels_set = set()
    for i in range(1, order + 1):
        for j in range(0, len(labels) - i + 1):
            labels_set.add(labels[j])
            order_gram = " ".join(reversed(labels[j:i + j]))
            if order_gram not in labels_count:
                labels_count[order_gram] = 1
            else:
                labels_count[order_gram] += 1
    return labels_count, labels_set


def label_word_pairs(words_and_labels):
    pairs_count = dict()
    word_count = dict()
    for word, label in words_and_labels:
        pair = word + " " + label
        if pair not in pairs_count:
            pairs_count[pair] = 1
        else:
            pairs_count[pair] += 1
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1

    return pairs_count, word_count


def calculate_UNK_probs(words_and_labels):
    first_half = words_and_labels[:len(words_and_labels) / 2]
    first_haft_words = set()
    second_half = words_and_labels[len(words_and_labels) / 2:]
    unk_dict = dict()
    for word, label in first_half:
        first_haft_words.add(word)
    for word, label in second_half:
        if word not in first_haft_words:
            word_pair = "UNK " + label
            if word_pair in unk_dict:
                unk_dict[word_pair] += 1
            else:
                unk_dict[word_pair] = 1
    return unk_dict


class ProbabilityContainer:
    def __init__(self, e_file_name, q_file_name, lambda_one=0.8, lambda_two=0.15, lambda_three=0.05):
        labels_word_count = read_file(e_file_name, parse_count_reading)
        labels_count = read_file(q_file_name, parse_count_reading)
        self._q = dict()
        self._e = dict()
        self._word_count = dict()
        self._label_set_count = dict()

        self._calculate_UNK_sigs(labels_word_count)
        self._label_set = set()
        self._lambda_one = lambda_one
        self._lambda_two = lambda_two
        self._lambda_three = lambda_three
        self._calculate_q_probs(labels_count)
        self._calculate_e_probs(labels_word_count, labels_count)

    def get_label_set(self):
        return self._label_set

    def _calculate_UNK_sigs(self, pairs_count):
        unk_sig_dict = dict()
        for word_label in pairs_count:
            word, label = word_label.split(" ")
            if word not in self._word_count:
                self._word_count[word] = pairs_count[word_label]
            else:
                self._word_count[word] += pairs_count[word_label]
            if label not in self._label_set_count:
                self._label_set_count[label] = pairs_count[word_label]
            else:
                self._label_set_count[label] += pairs_count[word_label]

            for suffix in SUFFIXES:
                if word.endswith(suffix):
                    end_pair = "UNK." + suffix + " " + label
                    if end_pair not in unk_sig_dict:
                        unk_sig_dict[end_pair] = pairs_count[word_label]
                    else:
                        unk_sig_dict[end_pair] += pairs_count[word_label]
            for prefix in PREFIXES:
                if word.endswith(prefix):
                    begin_pair = prefix + ".UNK" + " " + label
                    if begin_pair not in unk_sig_dict:
                        unk_sig_dict[begin_pair] = pairs_count[word_label]
                    else:
                        unk_sig_dict[begin_pair] += pairs_count[word_label]
            if word[0].isupper():
                begin_pair = "Upper.UNK " + label
                if begin_pair not in unk_sig_dict:
                    unk_sig_dict[begin_pair] = pairs_count[word_label]
                else:
                    unk_sig_dict[begin_pair] += pairs_count[word_label]
        #print unk_sig_dict
        pairs_count.update(unk_sig_dict)

    # Gets a dictionary of label keys and maps them to their count.
    def _calculate_q_probs(self, labels_count):
        labels_sum = sum(labels_count.values())
        for key in labels_count:
            tokens = key.split(" ")
            number_of_tokens = len(tokens)
            if number_of_tokens == 3:
                self._q[key] = float(labels_count[key]) / float(labels_count[" ".join(tokens[1:3])])
            elif number_of_tokens == 2:
                self._q[key] = float(labels_count[key]) / float(labels_count[tokens[-1]])
            else:
                self._label_set.add(key)
                self._q[key] = float(labels_count[key]) / float(labels_sum)

    # Gets a dictionary of label and word and maps them to their count.
    def _calculate_e_probs(self, label_word_count, labels_count):
        unk_values = dict()
        for key in label_word_count:
            word_label = key.split(" ")
            if word_label[0] == "UNK":
                unk_values[key] = label_word_count[key]
            else:
                self._e[key] = float(label_word_count[key]) / float(self._label_set_count[word_label[-1]])
        UNK_sum_values = sum(unk_values.values())
        for key in unk_values:
            self._e[key] = float(unk_values[key]) / float(UNK_sum_values)

    def _suffixes_prob(self, word, label):

        prob = 0
        for suffix in SUFFIXES:
            if word.endswith(suffix):
                end_pair = "UNK." + suffix + " " + label
                if end_pair in self._e:
                    return self._e[end_pair]
        return prob

    def _prefixes_prob(self, word, label):
        prob = 0
        for prefix in PREFIXES:
            if word.startswith(prefix):
                start_pair = prefix + ".UNK" + " " + label
                if start_pair in self._e:
                    prob = self._e[start_pair]
        return prob

    # Gets a word and a label and returns the LOG e probability for that word given that label.
    def get_e_prob(self, word, label):

        prob = 0
        key = word + " " + label
        epsilon = 1e-8
        if word in self._word_count and self._word_count[word] > 2:
            if key in self._e:
                prob = self._e[key]
            else:
                prob =  epsilon # 1.0 / float(len(self._label_set)) #TODO: maybe change the conditions?
        else:
            unk_label = "UNK " + label

            if label == "CD" and is_number(word): return 1.
            suffix_prob = self._suffixes_prob(word, label)
            prefix_prob = self._prefixes_prob(word, label)

            if unk_label in self._e:
                prob = 0.1 * self._e["UNK " + label] + 0.4 * suffix_prob + 0.4 * prefix_prob
            else:
                if prob == 0:
                    # guess a label randomely in a uniform manner
                    return epsilon #1.0 / float(len(self._label_set))
        return prob

    def get_score(self, word, tag, tag_prev, tag_prev_prev):
        q = self.get_q_prob(tag, tag_prev, tag_prev_prev)
        e = self.get_e_prob(word, tag)

        score = np.where(q > 0, np.log(q), -np.inf) + np.where(e > 0, np.log(e), -np.inf)
        return score

    """"
    Gets a label and returns the LOG q probability for that label (according to the conditional
    probabilities described in class).
    """

    def get_q_prob(self, y, t2, t1):
        """returns the probability p(y|t1 t2) (last two tags seen are t1 then t2)"""
        p1, p2, p3 = 0, 0, self._q[y]
        one_backwards = y + " " + t2
        two_backwards = one_backwards + " " + t1
        # Special case when looking at the first word.
        if t2 == "Start":
            if one_backwards in self._q:
                return self._q[one_backwards]
            else:
                return self._q[y]
        if one_backwards in self._q:
            p2 = self._q[one_backwards]
            if two_backwards in self._q:
                p3 = self._q[two_backwards]

        prob = self._lambda_one * p1 + self._lambda_two * p2 + self._lambda_three * p3
        assert prob >= 0 and prob <= 1
        return prob
