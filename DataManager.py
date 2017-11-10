def parse_pos_reading(lines):
    content = [["^^^^^/Start"] + line.split(" ")[0:-1] for line in lines]
    data = [word for line in content for word in line]
    return [word.rsplit("/", 1) for word in data]


def parse_pos_writing(counts):
    my_str = ""
    for key in counts.keys():
        my_str += key + '\t' + str(counts[key]) + '\n'
    return my_str


def read_file(file_name, parse_func):
    file = open(file_name, 'r')
    lines = file.readlines()
    file.close()
    return parse_func(lines)


def write_file(file_name, content, parse_func):
    file = open(file_name, 'w')
    file.write(parse_func(content))
    file.close()


def count_labels(labels, order):
    labels_count = dict()
    labels_set = set()
    for i in range(1, order + 1):
        for j in range(0, len(labels)):
            labels_set.add(labels[j])
            order_gram = " ".join(labels[j:i + j])
            if order_gram not in labels_count:
                labels_count[order_gram] = 1
            else:
                labels_count[order_gram] += 1
    return labels_count, labels_set


def label_word_pairs(words, labels):
    pairs_count = dict()
    for word, label in zip(words, labels):
        pair = word + " " + label
        if pair not in pairs_count:
            pairs_count[pair] = 1
        else:
            pairs_count[pair] += 1
    return pairs_count


class ProbabilityContainer:
    def __init__(self):
        self._q = dict()
        self._e = dict()

    def calculate_probs(self, labels_count, label_word_count):
        pass

    def get_e_probs(self, words, labels):
        raise NotImplementedError

    def get_q_probs(self, labels):
        raise NotImplementedError


words_and_labels = read_file("data/ass1-tagger-train", parse_pos_reading)
words = [data[0] for data in words_and_labels]
labels = [data[1] for data in words_and_labels]

labels_count, labels_set = count_labels(labels, 3)
write_file("q.mle", labels_count, parse_pos_writing)

pairs_count = label_word_pairs(words, labels)
write_file("e.mle", pairs_count, parse_pos_writing)
