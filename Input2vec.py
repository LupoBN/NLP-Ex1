import FeatureCreator
from sklearn.feature_extraction import DictVectorizer
import time


class Input2vec():
    def __init__(self, feature_map_filename):
        self.feature_map = open(feature_map_filename, "r")
        self.map = {}
        self._I2T = {}
        self.T2I = {}
        self.create_mapping()
        self.creator = FeatureCreator.FeatureCreator()
        self._vec = DictVectorizer(sparse=True)

        self._vec.fit_transform(self.map)
        print "map size is ", len(self.map)

    def ind_to_tag(self, index):
        return self._I2T[str(index)]

    def get_labels_set(self):
        return set(self._I2T.values())

    def create_mapping(self):
        lines = self.feature_map.readlines()
        for line in lines:
            line = line.replace("\n", "")
            key, value = line.split("\t")
            if "=" not in key:
                self._I2T[value] = key
                self.T2I[key] = value
            else:
                self.map[key] = int(value)

    def create_vector(self, word, p, pp, nw, nnw):
        """

        :param word: current word
        :param p:  a tuple [prev_word, prev_label]
        :param pp: a tuple [prev_prev_word, prev_prev_label]
        :param nw: next word
        :param nnw:next next word
        :return:
        """
        vector = self.creator.create_features(word, p, pp, nw, nnw, True, True)
        print vector
        features = vector.strip().split(" ")
        print len(features)
        feature_vec = dict()
        for f in features:
            feature_vec[f] = 1
        x = self._vec.transform(feature_vec)

        # sorted_vector = sorted(features_vector, key = lambda pair: int(pair[:pair.index(":")]))
        # rtrn_val = " ".join(sorted_vector)
        return x


if __name__ == '__main__':
    word = "are"
    p = ["how", "DT"]
    pp = ["hello", "JJ"]
    nw = "you"  # NOTE: if there is no next word, i.e. it's the end of the sentence, set nw, nnw=""
    nnw = "?"
    inp2vec = Input2vec("feature_map_file")
    r =  inp2vec.create_vector(word, p, pp, nw, nnw)
    print r.size
