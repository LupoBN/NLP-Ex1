import sys
import TrainSolver
from ViterbiTagger import ViterbiTagger
from Helpers import test_model

if __name__ == '__main__':
    llm = TrainSolver.LogLinearModel()
    llm.load_model(sys.argv[2])
    vt = ViterbiTagger(llm)
    test_model("ass1-tagger-test", vt)
