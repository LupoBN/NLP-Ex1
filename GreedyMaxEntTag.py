import sys
import TrainSolver
from GreedyTagger import GreedyTagger
from Helpers import test_model

if __name__ == '__main__':
    llm = TrainSolver.LogLinearModel()
    llm.load_model(sys.argv[2])
    gt = GreedyTagger(llm)
    test_model("ass1-tagger-test", gt)
