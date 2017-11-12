import numpy as np

SUFFIXES = ["ing", "ed", "s"]
PREFIXES = ["un", "de", "re"]


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


def parse_possible_labels_writing(possible_labels):
    my_str = ""
    for key in possible_labels.keys():
        my_str += key
        for label in possible_labels[key]:
            my_str += '\t' + label
        my_str += '\n'
    return my_str


def parse_possible_labels(word_and_labels):
    """returns a dictionary possible_labels where possible_labels[word] is a list
    of all labels encountered in the training set for that word. used for viterbi optimization. """
    possible_labels = {}
    for i, word_label in enumerate(word_and_labels):
        word, label = word_label
        if not word in possible_labels:
            possible_labels[word] = set([label])
        else:
            possible_labels[word].add(label)
    return possible_labels


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
        for j in range(0, len(labels) - i + 1):
            labels_set.add(labels[j])
            order_gram = " ".join(reversed(labels[j:i + j]))
            if order_gram not in labels_count:
                labels_count[order_gram] = 1
            else:
                labels_count[order_gram] += 1
    return labels_count, labels_set


def label_UNK_sig(word, label, pairs_count):
    for suffix in SUFFIXES:
        if word.endswith(suffix):
            end_pair = "UNK." + suffix + " " + label
            if end_pair not in pairs_count:
                pairs_count[end_pair] = 1
            else:
                pairs_count[end_pair] += 1
    for prefix in PREFIXES:
        if word.endswith(prefix):
            begin_pair = prefix + ".UNK" + " " + label
            if begin_pair not in pairs_count:
                pairs_count[begin_pair] = 1
            else:
                pairs_count[begin_pair] += 1
    if word[0].isupper():
        begin_pair = "Upper.UNK " + label
        if begin_pair not in pairs_count:
            pairs_count[begin_pair] = 1
        else:
            pairs_count[begin_pair] += 1


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

        label_UNK_sig(word, label, pairs_count)
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
    def __init__(self, labels_count, labels_word_count, word_count, lambda_one=0.8, lambda_two=0.15, lambda_three=0.05):
        self._q = dict()
        self._e = dict()
        self._label_set = set()
        self._word_count = word_count
        self._lambda_one = lambda_one
        self._lambda_two = lambda_two
        self._lambda_three = lambda_three
        self._calculate_q_probs(labels_count)
        self._calculate_e_probs(labels_word_count, labels_count)

    # Gets a dictionary of label keys and maps them to their count.

    def _calculate_q_probs(self, labels_count):
        for key in labels_count:
            tokens = key.split(" ")
            number_of_tokens = len(tokens)
            if number_of_tokens == 3:
                self._q[key] = float(labels_count[key]) / float(labels_count[" ".join(tokens[1:-1])])
            elif number_of_tokens == 2:
                self._q[key] = float(labels_count[key]) / float(labels_count[tokens[-1]])
            else:
                self._label_set.add(key)
                self._q[key] = float(labels_count[key]) / float(len(labels_count))

    # Gets a dictionary of label and word and maps them to their count.
    def _calculate_e_probs(self, label_word_count, labels_count):
        unk_values = dict()
        for key in label_word_count:
            word_label = key.split(" ")
            if word_label[0] == "UNK":
                unk_values[key] = label_word_count[key]
            else:
                self._e[key] = float(label_word_count[key]) / float(labels_count[word_label[-1]])
        UNK_sum_values = sum(unk_values.values())
        for key in unk_values:
            self._e[key] = float(unk_values[key]) / float(UNK_sum_values)

    def _suffixes_prob(self, word, label):
        prob = 0
        for suffix in SUFFIXES:
            if word.endswith(suffix):
                end_pair = "UNK." + suffix + " " + label
                if end_pair in self._e:
                    return np.log(self._e[end_pair])
        return prob

    def _prefixes_prob(self, word, label):
        prob = 0
        for prefix in PREFIXES:
            if word.startswith(prefix):
                start_pair = prefix + ".UNK" + " " + label
                if start_pair in self._e:
                    prob = np.log(self._e[start_pair])
        return prob

    # Gets a word and a label and returns the LOG e probability for that word given that label.
    def get_e_prob(self, word, label):
        prob = 0
        key = word + " " + label
        if key in self._word_count and self._word_count[key] > 2:
            prob = np.log(self._e[key])
        else:
            unk_label = "UNK " + label
            suffix_prob = self._suffixes_prob(word, label)
            prefix_prob = self._prefixes_prob(word, label)
            if unk_label in self._e:
                prob = 0.7 * np.log(self._e["UNK " + label]) + 0.15 * suffix_prob + 0.15 * prefix_prob
            else:
                if prob == 0:
                    return np.log(1.0 / float(len(self._label_set)))
        return prob

    """"
    Gets a label and returns the LOG q probability for that label (according to the conditional
    probabilities described in class).
    """

    def get_q_prob(self, y, t1, t2):
        p1, p2, p3 = 0, 0, np.log(self._q[y])
        one_backwards = y + " " + t1
        two_backwards = one_backwards + " " + t2
        # Special case when looking at the first word.
        if t1 == "Start":
            return np.log(self._q[one_backwards])
        if one_backwards in self._q:
            p2 = np.log(self._q[one_backwards])
            if two_backwards in self._q:
                p3 = np.log(self._q[two_backwards])
        return self._lambda_one * p1 + self._lambda_two * p2 + self._lambda_three * p3
