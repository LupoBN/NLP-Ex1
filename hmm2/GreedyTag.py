import sys

sys.path.insert(0, '../Extras')
import DataManager
from Helpers import test_model
from Taggers import GreedyTagger
from ProbabilitiyContainers import ProbabilityContainer

if __name__ == '__main__':
    probability_provider = ProbabilityContainer(sys.argv[3], sys.argv[2])
    gt = GreedyTagger(probability_provider)

    print test_model(sys.argv[1], gt)
