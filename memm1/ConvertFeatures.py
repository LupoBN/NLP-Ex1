import sys

sys.path.insert(0, '../Extras')


class FeaturesConverter():
    def __init__(self, features_set, labels_set, features2numbers, labels2numbers):
        self.features_set = features_set
        self.labels_set = labels_set
        self.features2numbers = features2numbers
        self.labels2numbers = labels2numbers

    def convert(self, lines):

        s = ""
        ss = set()
        for j, line in enumerate(lines):

            words = line.replace("\n", "").split(" ")
            if "" in words: print "test: ", "" in words
            for i, word in enumerate(words):
                if i == 0:
                    val = self.labels2numbers[word]
                    ss.add(word)
                    s += str(val) + " "
                else:
                    val = self.features2numbers[word]
                    s += str(val) + ":1 "

            s += "\n"

        lines = s.strip().split("\n")
        # print "last is :" ,lines[-1]
        for i, line in enumerate(lines):
            splitted = line.strip().split(" ")
            label, rest = splitted[0], splitted[1:]
            # if "" in rest: rest.remove("")
            rest = sorted(rest, key=lambda pair: int(
                pair[:pair.index(":")]))  # sort each line according to the number encoding the feature
            lines[i] = " ".join([label] + rest) + "\n"

        q = "".join(lines)
        f = open(sys.argv[2], "w")
        f.write(q)
        f.close()



if __name__ == '__main__':
    f = open(sys.argv[1])
    lines = f.readlines()
    labels_set = set([line.split(" ")[0] for line in lines])
    features_set = set(("".join(lines)).replace("\n", " ").strip().split(" "))
    for l in labels_set: features_set.remove(l)

    features2numbers = {f: i for i, f in enumerate(sorted(features_set))}
    numbers2features = {i: f for i, f in enumerate(sorted(features_set))}

    labels2numbers = {l: i for i, l in enumerate(sorted(labels_set))}
    numbers2labels = {i: l for i, l in enumerate(sorted(labels_set))}

    f = open(sys.argv[3], "w")
    for key in sorted(labels2numbers, key = lambda key: labels2numbers[key]):
        val = labels2numbers[key]
        f.write(str(key) + "\t" + str(val) + "\n")

    for key in sorted(features2numbers, key=lambda key: features2numbers[key]):
        val = features2numbers[key]
        f.write(str(key) + "\t" + str(val) + "\n")
    f.close()

fc = FeaturesConverter(features_set, labels_set, features2numbers, labels2numbers)
fc.convert(lines)
