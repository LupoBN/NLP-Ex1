import DataManager


class GreedyTagger:
    def __init__(self, e_count, q_count):
        self.e_count = e_count
        self.q_count = q_count
        self.labels_set = DataManager.labels_set

    def get_count(self, dictionary, key):
        if (key in dictionary.keys()):
            return dictionary[key]
        else:
            return 0


    def predict_tags(self, file_name):
        words_and_labels, words, labels = DataManager.DataManager.read_file(file_name)
        labels = set(labels)
        predictons = []
        for word in words:
            best_label = None
            best_prob = -1.
            for i, label in enumerate(self.labels_set):
                word_label = word + " " + label
                if i >= 2:
                    last_two = " ".join([predictons[-2], predictons[-1]])
                    q_nom = self.get_count(self.q_count, last_two + " " + label)
                    if q_nom == 0:
                        q = 0.
                    else:
                        q = q_nom / (1. * self.q_count[last_two])
                else:
                    q = self.q_count[label]
                e = self.e_count[word_label] / (1. * self.e_count[label])
                if q * e > best_prob:
                    best_prob = q * e
                    best_label = label
                    predictons.add(best_label)
        return predictons


gt = GreedyTagger(DataManager.pairs_count, DataManager.labels_count)
gt.predict_tags("data/ass1-tagger-train")

i = 5
