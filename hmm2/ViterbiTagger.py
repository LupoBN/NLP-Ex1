import sys

sys.path.insert(0, '../Extras')
import DataManager
from Helpers import test_model
from Algorithms import ViterbiTagger

if __name__ == '__main__':
    words_and_labels = DataManager.read_file("data/ass1-tagger-train", DataManager.parse_pos_reading)
    possible_labels = DataManager.parse_possible_labels(words_and_labels)

    probability_provider = DataManager.ProbabilityContainer(sys.argv[3], sys.argv[2])

    vt = ViterbiTagger(probability_provider, possible_labels)
    print test_model(sys.argv[1], vt)
