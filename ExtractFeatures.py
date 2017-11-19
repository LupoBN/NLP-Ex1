import DataManager
import sys


def contains_number(s):
    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    for d in digits:
        if d in s:
            return True
    return False


class FeatureExtractor():
    def __init__(self, rare_words, word_set, words_and_labels, labels_set):
        self.rare_words = rare_words
        self.common_words = set([word for word in word_set if word not in rare_words])
        self.words_set = word_set
        self.words_and_labels = words_and_labels
        self.labels_set = labels_set

    def create_features(self, word, p, pp, nw, nnw):
        """

        :param word: current word
        :param label: current label
        :param p: prev (word, label) tuple
        :param pp: prev prev (word, label) tuple
        :return:
        """

        pt, ppt = p[1], pp[1]
        pw, ppw = p[0], pp[0]
        s = ""

        if word in self.common_words:

            s += "form=" + word + " "

        else:
            s += self._create_prefix_features(word)
            s += self._create_suffix_features(word)
            s += self.create_inner_chars_features(word)

        s += "pt=" + pt + " "
        s += "ppt_pt=" + ppt + "_" + pt + " "
        s += "pw=" + pw + " "
        s += "ppw=" + ppw + " "
        s += "nw=" + nw + " "
        s += "nnw=" + nnw
        return s

    def _create_prefix_features(self, word):
        s = ""
        for i in range(1, min(5, len(word))):
            pref = word[:i]
            s += "pref=" + pref + " "
        return s

    def _create_suffix_features(self, word):
        s = ""
        for i in range(1, min(5, len(word))):
            suffix = word[-i:]
            s += "suf=" + suffix + " "
        return s

    def create_inner_chars_features(self, word):
        s = ""
        if contains_number(word):
            s += "number=True "
        if word[0].isupper():
            s += "upper=True "
        if "-" in word:
            s += "hyphen=True "
        # if len(s) > 0: s+=" "
        return s

    def extract(self, results_filename):
        s = ""
        l = len(self.words_and_labels)
        for i, (word, label) in enumerate(self.words_and_labels):
            if i < 2 or i > l - 3: continue
            s += label + " "
            p, pp = self.words_and_labels[i - 1], self.words_and_labels[i - 2]
            nw, nnw = self.words_and_labels[i + 1][0], self.words_and_labels[i + 2][0]
            s += self.create_features(word, p, pp, nw, nnw)
            # s = s.strip()
            s += "\n"

        features = set(s.replace("\n", " ").split(" "))
        features = set(filter(lambda word: "=" in word, features))  # filter out the gold labels

        print ("size is ", sys.getsizeof(features) / 1e6, " mb")

        f = open(results_filename, "w")
        f.write(s)
        f.close()

        l = list(features)
        for i in l[:100]:
            print i

        return features


if __name__ == '__main__':
    words_and_labels = DataManager.read_file("data/ass1-tagger-train", DataManager.parse_pos_reading)
    labels = [data[1] for data in words_and_labels]
    words = [data[0] for data in words_and_labels]
    word_set = set(words)
    labels_count, labels_set = DataManager.count_labels(labels, 1)
    pairs_count, word_count = DataManager.label_word_pairs(words_and_labels)
    rare_words = set([word for word in word_set if word_count[word] < 5])

    fe = FeatureExtractor(rare_words, word_set, words_and_labels, labels_set)
    fe.extract("train_features")
    # for w, l in words_and_labels[:200]:
    #    print w,l
