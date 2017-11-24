from DataManager import *
from sklearn import linear_model
import pickle

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
        # print unk_sig_dict
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
                prob = epsilon  # 1.0 / float(len(self._label_set)) #TODO: maybe change the conditions?
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
                    return epsilon  # 1.0 / float(len(self._label_set))
        return prob

    def get_score(self, words, tag, tag_prev, tag_prev_prev):
        q = self.get_q_prob(tag, tag_prev, tag_prev_prev)
        e = self.get_e_prob(words[0], tag)
        if q > 0:
            q = np.log(q)
        else:
            q = -np.inf
        if e > 0:
            e = np.log(e)
        else:
            e = -np.inf
        score = q + e
        return score

    def get_probabilities(self, words, tag_prev, tag_prev_prev):
        probabilities = dict()
        for tag in self._label_set:
            q = self.get_q_prob(tag, tag_prev, tag_prev_prev)
            e = self.get_e_prob(words[0], tag)
            probabilities[tag] = np.log(q) + np.log(e)
        return probabilities

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
        if t2 == START_TAG:
            if one_backwards in self._q:
                return self._q[one_backwards]
            else:
                return self._q[y]
        if one_backwards in self._q:
            p2 = self._q[one_backwards]
            if two_backwards in self._q:
                p3 = self._q[two_backwards]

        prob = self._lambda_one * p1 + self._lambda_two * p2 + self._lambda_three * p3
        # assert prob >= 0 and prob <= 1
        return prob

class LogLinearModel:
    def __init__(self, I2V=None, file_name=None):
        if file_name is None:
            self._model = linear_model.LogisticRegression()
        else:
            self._model = pickle.load(open(file_name, 'rb'))
        self._I2V = I2V

    def train_model(self, x_train, y_train):
        self._model = self._model.fit(x_train, y_train)

    def get_label_set(self):
        return self._I2V.get_labels_set()

    def get_probabilities(self, words, tag_prev, tag_prev_prev):

        x = self._I2V.create_vector(words[0], [words[1], tag_prev],
                                    [words[2], tag_prev_prev], words[3], words[4])
        probs = self._model.predict_log_proba(x)[0]
        prob_dict = {self._I2V.ind_to_tag(str(i)): probs[i] for i in range(0, probs.size)}
        return prob_dict

    def get_score(self, words, tag, tag_prev, tag_prev_prev):
        x = self._I2V.create_vector(words[0], [words[1], tag_prev],
                                    [words[2], tag_prev_prev], words[3], words[4])
        probs = self._model.predict_log_proba(x)[0]
        score = probs[int(self._I2V.T2I[tag])]
        return score

    def save_model(self, file_name):
        pickle.dump(self._model, open(file_name, 'wb'))

    def load_model(self, file_name):
        self._model = pickle.load(open(file_name, 'rb'))

    def test_model(self, x_test, y_test):
        return self._model.score(x_test, y_test)
