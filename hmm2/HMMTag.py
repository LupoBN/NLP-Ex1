import sys

sys.path.insert(0, '../Extras')
from Helpers import *
from ProbabilitiyContainers import ProbabilityContainer
from Taggers import ViterbiTagger

if __name__ == '__main__':
    words_and_labels = read_file("../data/ass-tagger-train", parse_pos_reading)
    possible_labels = parse_possible_labels(words_and_labels)

    probability_provider = ProbabilityContainer(sys.argv[3], sys.argv[2])

    vt = ViterbiTagger(probability_provider, possible_labels)
    print test_model(sys.argv[1], vt)
