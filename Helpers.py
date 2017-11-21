import time
def test_model(file_name, model):
    test_file = open(file_name)
    lines = test_file.readlines()

    good, bad = 0., 0.
    n = len(lines)
    print n
    for i in range(n):
        print str(i)
        words_orig = ("^^^^^/Start " + lines[i]).split(" ")
        words = [word.split("/")[0] for word in words_orig]
        labels = [word.split("/")[1].strip("\n") for word in words_orig]
        preds = model.predict_tags(words)
        s = ""
        for i, word in enumerate(words[1:]):

            s += word + "(" + preds[i] + ") "
            if preds[i] == labels[i + 1]:
                good += 1
            else:

                bad += 1
    test_file.close()


    return (good) / (good + bad)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_pos_reading(lines):
    content = [["^^^^^/Start"] + line.split(" ")[0:-1] for line in lines]
    data = [word for line in content for word in line]
    return [word.rsplit("/", 1) for word in data]

def parse_map_reading(lines):
    pairs_count = {line.split("\t")[0]: int(line.split("\t")[1]) for line in lines}
    return pairs_count



def parse_count_reading(lines):
    pairs_count = {line.split("\t")[0]: float(line.split("\t")[1]) for line in lines}
    return pairs_count


def parse_count_reading(lines):
    pairs_count = {line.split("\t")[0]: float(line.split("\t")[1]) for line in lines}
    return pairs_count


def parse_pos_writing(counts):
    my_str = ""
    for key in counts.keys():
        my_str += key + '\t' + str(counts[key]) + '\n'
    return my_str


def parse_possible_labels_writing(possible_labels):
    my_str = ""
    for key in possible_labels.keys():
        my_str += key
        for label in possible_labels[key]:
            my_str += '\t' + label
        my_str += '\n'
    return my_str


def parse_possible_labels(word_and_labels):
    """returns a dictionary possible_labels where possible_labels[word] is a list
    of all labels encountered in the training set for that word. used for viterbi optimization. """
    possible_labels = {}
    for i, word_label in enumerate(word_and_labels):
        word, label = word_label
        if not word in possible_labels:
            possible_labels[word] = set([label])
        else:
            possible_labels[word].add(label)
    return possible_labels


def read_file(file_name, parse_func):
    file = open(file_name, 'r')
    lines = file.readlines()
    file.close()
    return parse_func(lines)


def write_file(file_name, content, parse_func):
    file = open(file_name, 'w')
    file.write(parse_func(content))
    file.close()
