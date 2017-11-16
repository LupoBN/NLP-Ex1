
class FeaturesConverter():
    def __init__(self, features_set, labels_set, features2numbers, labels2numbers):
        self.features_set = features_set
        self.labels_set = labels_set
        self.features2numbers = features2numbers
        self.labels2numbers = labels2numbers


    def convert(self, lines):
        "feature_vecs_file"

        s = ""
        for line in lines:
            words = line.replace("\n","").split(" ")
            for i, word in enumerate(words):
                if i == 0:
                    val = self.labels2numbers[word]
                    s += str(val)+" "
                else:
                    val = self.features2numbers[word]
                    s += str(val) + ":1 "
            s+="\n"

        f = open("feature_vecs_file", "w")
        f.write(s)
        f.close()



if __name__ == '__main__':
 f = open("train_features")
 lines = f.readlines()
 labels_set = set([line.split(" ")[0] for line in lines])
 features_set = set(  ("".join(lines)).replace("\n", " ").split(" ")   )

 features2numbers = {f:i for i,f in enumerate(sorted(features_set))}
 numbers2features = {i:f for i, f in enumerate(sorted(features_set))}
 labels2numbers = {l:i for i,l in enumerate(sorted(labels_set))}
 numbers2labels = {i:l for i, l in enumerate(sorted(labels_set))}

 f = open("feature_map_file", "w")
 for key, val in features2numbers.iteritems():
     f.write(str(key)+"\t"+str(val)+"\n")
 f.close()

 fc = FeaturesConverter(features_set, labels_set, features2numbers, labels2numbers)
 fc.convert(lines)

