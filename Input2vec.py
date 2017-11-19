import FeatureCreator


class Input2vec():
    def __init__(self, feature_map_filename):
        self.feature_map = open(feature_map_filename, "r")
        self.map = {}
        self.create_mapping()
        self.creator = FeatureCreator.FeatureCreator()

    def create_mapping(self):
        lines = self.feature_map.readlines()
        for line in lines:
            line = line.replace("\n","")
            key, value = line.split("\t")
            self.map[key] = value

    def create_vector(self,  word, p, pp, nw, nnw):
        """

        :param word: current word
        :param p:  a tuple [prev_word, prev_label]
        :param pp: a tuple [prev_prev_word, prev_prev_label]
        :param nw: next word
        :param nnw:next next word
        :return:
        """
        features_vector = ""
        vector = self.creator.create_features( word, p, pp, nw, nnw, True, True)
        features = vector.split(" ")
        for f in features:
            if f in self.map:
                encoded_feature = self.map[f]
                features_vector+=str(encoded_feature)+":1 "

        features_vector = features_vector.strip().split(" ")
        sorted_vector = sorted(features_vector, key = lambda pair: int(pair[:pair.index(":")]))
        rtrn_val = " ".join(sorted_vector)
        return " ".join(sorted_vector)



if __name__ == '__main__':
    word = "are"
    p = ["how", "DT"]
    pp = ["hello", "JJ"]
    nw = "you" #NOTE: if there is no next word, i.e. it's the end of the sentence, set nw, nnw=""
    nnw = "?"
    inp2vec = Input2vec("feature_map_file")
    print inp2vec.create_vector(word, p, pp, nw, nnw)