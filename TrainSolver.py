import sys
from sklearn import linear_model
from sklearn.datasets import load_svmlight_file
import pickle

class LogLinearModel:
    def __init__(self):
        self._model = linear_model.LogisticRegression()
        pass

    def train_model(self, x_train, y_train):
        self._model = self._model.fit(x_train, y_train)

    def score(self, word, tag, tag_prev, tag_prev_prev):
        #TODO: Make x a feature vector according to the word and tags given.
        x = None
        return self._model.predict_proba(x)


    def save_model(self, file_name):
        pickle.dump(self._model, open(file_name, 'wb'))

    def load_model(self, file_name):
        self._model = pickle.load(open(file_name, 'rb'))


if __name__ == '__main__':
    x_train, y_train = load_svmlight_file(sys.argv[1])
    trainer = LogLinearModel()
    trainer.train_model(x_train, y_train)
    trainer.save_model("model")
    pass
