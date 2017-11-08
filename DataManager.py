# TODO: Split might loss some numbers between them are 5/100 (5 divided by 100).
class DataManager:
    def __init__(self):

    def read_file(self, file_name):
        file = open(file_name, 'r')
        lines = file.readlines()
        lines = [line.split(" ")[0:-1] for line in lines]
        data = [word for line in lines for word in line]
        words_and_labels = [["".join(word.split("/")[0:-1]), word.split("/")[-1]] for word in data]
        words = [data[0] for data in words_and_labels]
        labels = [data[1] for data in words_and_labels]

        file.close()

        assert len(words) == len(labels)
        return words_and_labels, words, labels

    def count_labels(labels, order):
        labels_count = dict()
        labels_set = set()
        for j in range(1, order + 1):
            for i in range(0, len(labels)):
                labels_set.add(labels[i])
                order_gram = " ".join(labels[i:i + j])
                if (order_gram not in labels_count.keys()):
                    labels_count[order_gram] = 0
                else:
                    labels_count[order_gram] += 1
        return labels_count, labels_set

    def label_word_pairs(words, labels):
        pairs_count = {}
        for word, label in zip(words, labels):
            pair = word + " " + label
            if pair not in pairs_count.keys():
                pairs_count[pair] = 0
            else:
                pairs_count[pair] += 1

        return pairs_count

    def write_statistics(statistics, file_name):
        file = open(file_name, 'w')
        for key in statistics.keys():
            file.write(key + '\t' + str(statistics[key]) + '\n')
        file.close()

    def get_e_probs(words, labels):
        raise NotImplementedError

    def get_q_probs(labels):
        raise NotADirectoryError


words_and_labels, words, labels = DataManager.read_file("data/ass1-tagger-train")
labels_count, labels_set = DataManager.count_labels(labels, 3)
DataManager.write_statistics(labels_count, "q.mle")
pairs_count = DataManager.label_word_pairs(words, labels)
DataManager.write_statistics(pairs_count, 'e.mle')
