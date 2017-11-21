import sys
import TrainSolver
from GreedyTagger import GreedyTagger
from Helpers import test_model
from Input2vec import Input2vec
if __name__ == '__main__':
    i2v = Input2vec(sys.argv[3])
    llm = TrainSolver.LogLinearModel(i2v)
    llm.load_model(sys.argv[2])
    gt = GreedyTagger(llm)
    print test_model(sys.argv[1], gt)

