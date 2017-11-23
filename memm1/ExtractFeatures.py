import sys

sys.path.insert(0, '../Extras')
import DataManager
import FeatureCreator
import sys





class FeatureExtractor():
    def __init__(self, rare_words, word_set, words_and_labels):
        self.rare_words = rare_words
        self.common_words = set([word for word in word_set if word not in rare_words])
        self.words_set = word_set
        self.words_and_labels = words_and_labels
        self.creator = FeatureCreator.FeatureCreator()

    def extract(self, results_filename):
        s = ""
        l = len(self.words_and_labels)
        for i, (word, label) in enumerate(self.words_and_labels):
            if i < 2 or i > l - 3: continue
            s += label + " "
            p, pp = self.words_and_labels[i - 1], self.words_and_labels[i - 2]
            nw, nnw = self.words_and_labels[i + 1][0], self.words_and_labels[i + 2][0]
            isCommon = word in self.common_words
            s += self.creator.create_features(word, p, pp, nw, nnw, isCommon) # self.create_features(word, p, pp, nw, nnw)
            # s = s.strip()
            s += "\n"

        features = set(s.replace("\n", " ").split(" "))
        features = set(filter(lambda word: "=" in word, features))  # filter out the gold labels

        #print ("size is ", sys.getsizeof(features) / 1e6, " mb")

        f = open(results_filename, "w")
        f.write(s)
        f.close()

        return features


if __name__ == '__main__':
    words_and_labels = DataManager.read_file(sys.argv[1], DataManager.parse_pos_reading)

    labels = [data[1] for data in words_and_labels]
    words = [data[0] for data in words_and_labels]
    word_set = set(words)
    pairs_count, word_count = DataManager.label_word_pairs(words_and_labels)
    rare_words = set([word for word in word_set if word_count[word] < 5])

    fe = FeatureExtractor(rare_words, word_set, words_and_labels)
    fe.extract(sys.argv[2])

