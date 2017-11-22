import sys

sys.path.insert(0, '../Extras')
from ProbabilitiyContainers import LogLinearModel
from Taggers import ViterbiTagger
from Helpers import *
from Input2vec import Input2vec

if __name__ == '__main__':
    i2v = Input2vec(sys.argv[3])
    words_and_labels = read_file("../data/ass-tagger-train", parse_pos_reading)
    possible_labels = parse_possible_labels(words_and_labels)
    llm = LogLinearModel(i2v, sys.argv[2])
    vt = ViterbiTagger(llm, possible_labels)
    print test_model(sys.argv[1], vt)
