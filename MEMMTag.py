import sys
import TrainSolver
from ViterbiTagger import ViterbiTagger
from Helpers import test_model
from Input2vec import Input2vec
import DataManager
if __name__ == '__main__':
    i2v = Input2vec(sys.argv[3])
    words_and_labels = DataManager.read_file("data/ass1-tagger-train", DataManager.parse_pos_reading)
    possible_labels = DataManager.parse_possible_labels(words_and_labels)
    llm = TrainSolver.LogLinearModel(i2v, sys.argv[2])
    vt = ViterbiTagger(llm, possible_labels)
    print test_model(sys.argv[1], vt)
