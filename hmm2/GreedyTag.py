import sys

sys.path.insert(0, '../Extras')
from Helpers import write_prediction_file
from Taggers import GreedyTagger
from ProbabilitiyContainers import ProbabilityContainer

if __name__ == '__main__':
    probability_provider = ProbabilityContainer(sys.argv[3], sys.argv[2])
    gt = GreedyTagger(probability_provider)
    write_prediction_file(sys.argv[1], gt, sys.argv[4])
