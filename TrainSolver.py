import sys
from sklearn import linear_model
from sklearn.datasets import load_svmlight_file
import pickle


class LogLinearModel:
    def __init__(self, I2V = None, file_name = None):
        if file_name is None:
            self._model = linear_model.LogisticRegression(C=0.1)
        else:
            self._model = pickle.load(open(file_name, 'rb'))
        self._I2V = I2V

    def train_model(self, x_train, y_train):
        self._model = self._model.fit(x_train, y_train)

    def get_label_set(self):
        return self._I2V.get_labels_set()

    def get_probabilities(self, words, tag_prev, tag_prev_prev):

        x = self._I2V.create_vector(words[0], [words[1], tag_prev],
                                    [words[2], tag_prev_prev], words[3], words[4])
        probs = self._model.predict_log_proba(x)[0]
        prob_dict = {self._I2V.ind_to_tag(str(i)): probs[i] for i in range(0, probs.size)}
        return prob_dict


    def get_score(self, words, tag, tag_prev, tag_prev_prev):
        x = self._I2V.create_vector(words[0], [words[1], tag_prev],
                                    [words[2], tag_prev_prev], words[3], words[4])
        probs = self._model.predict_log_proba(x)[0]
        score = probs[int(self._I2V.T2I[tag])]
        return score

    def save_model(self, file_name):
        pickle.dump(self._model, open(file_name, 'wb'))

    def load_model(self, file_name):
        self._model = pickle.load(open(file_name, 'rb'))

    def test_model(self, x_test, y_test):
        return self._model.score(x_test, y_test)


if __name__ == '__main__':
    x_train, y_train = load_svmlight_file(sys.argv[1])
    llm = LogLinearModel()
    llm.train_model(x_train, y_train)
    llm.save_model("model")
    print "Training accuracy:", llm.test_model(x_train, y_train)

