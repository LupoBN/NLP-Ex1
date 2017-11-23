import sys

sys.path.insert(0, '../Extras')
from ProbabilitiyContainers import LogLinearModel
from Algorithms import GreedyTagger
from Helpers import write_prediction_file
from Input2vec import Input2vec

if __name__ == '__main__':
    i2v = Input2vec(sys.argv[3])
    llm = LogLinearModel(i2v, sys.argv[2])
    gt = GreedyTagger(llm)
    write_prediction_file(sys.argv[1], gt, sys.argv[4])
