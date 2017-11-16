import DataManager
import ExtractFeatures
import ConvertFeatures
import sklearn
from sklearn.datasets import load_svmlight_file

class Trainer():

    def __init__(self):
       pass


if __name__ == '__main__':
 f = open("train_features", "r")
 lines = f.readlines()
 X_train, y_train = load_svmlight_file("feature_vecs_file")
 print (X_train).getnnz()
