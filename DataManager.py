import numpy as np

def parse_pos_reading(lines):
    content = [["^^^^^/Start"] + line.split(" ")[0:-1] for line in lines]
    data = [word for line in content for word in line]
    return [word.rsplit("/", 1) for word in data]


def parse_count_reading(lines):
    pairs_count = {line.split("\t")[0]: float(line.split("\t")[1]) for line in lines}
    return pairs_count


def parse_pos_writing(counts):
    my_str = ""
    for key in counts.keys():
        my_str += key + '\t' + str(counts[key]) + '\n'
    return my_str


def read_file(file_name, parse_func):
    file = open(file_name, 'r')
    lines = file.readlines()
    file.close()
    return parse_func(lines)


def write_file(file_name, content, parse_func):
    file = open(file_name, 'w')
    file.write(parse_func(content))
    file.close()


def count_labels(labels, order):
    labels_count = dict()
    labels_set = set()
    for i in range(1, order + 1):
        for j in range(0, len(labels)):
            labels_set.add(labels[j])
            order_gram = " ".join(labels[j:i + j])
            if order_gram not in labels_count:
                labels_count[order_gram] = 1
            else:
                labels_count[order_gram] += 1
    return labels_count, labels_set


def label_word_pairs(words, labels):
    pairs_count = dict()
    for word, label in zip(words, labels):
        pair = word + " " + label
        if pair not in pairs_count:
            pairs_count[pair] = 1
        else:
            pairs_count[pair] += 1
    return pairs_count


class ProbabilityContainer:
    def __init__(self):
        self._q = dict()
        self._e = dict()

    def calculate_q_probs(self, labels_count):
        for key in labels_count:
            tokens = key.split(" ")
            number_of_tokens = len(tokens)
            if number_of_tokens == 3:
                self._q[key] = labels_count[key] / labels_count[(" ".join(tokens[0:1]))]
            elif number_of_tokens == 2:
                self._q[key] = labels_count[key] / labels_count[(" ".join(tokens[0:1]))]
            else:
                self._q[key] = labels_count[key] / len(labels_count)

    def calculate_e_probs(self, label_word_count, labels):
        for key in label_word_count:
            self._e[key] = label_word_count[key] / labels[key.split(" ")[-1]]

    # Returns the LOG e probability for word given a label.
    def get_e_prob(self, word, label):
        key = word + " " + label
        if key not in self._e:
            return 0
        return np.log(self._e[key])

    # Returns the LOG q probability for a label.
    def get_q_prob(self, label):
        if label not in self._q:
            return 0
        return np.log(self._q[label])
