import sys

sys.path.insert(0, '../Extras')

from sklearn.datasets import load_svmlight_file
from ProbabilitiyContainers import LogLinearModel


if __name__ == '__main__':
    x_train, y_train = load_svmlight_file(sys.argv[1])
    llm = LogLinearModel()
    llm.train_model(x_train, y_train)
    llm.save_model(sys.argv[2])
    #print "Training accuracy:", llm.test_model(x_train, y_train)
