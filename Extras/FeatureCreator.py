
class FeatureCreator():
    def __init__(self):
        pass

    def create_features(self, word, p, pp, nw, nnw, isCommon, createAll=False):
        """

        :param word: current word
        :param label: current label
        :param p: prev (word, label) tuple
        :param pp: prev prev (word, label) tuple
        :return:
        """

        pt, ppt = p[1], pp[1]
        pw, ppw = p[0], pp[0]
        s = ""

        if isCommon or createAll:

            s += "form=" + word + " "

        if not isCommon or createAll:
            s += self._create_prefix_features(word)
            s += self._create_suffix_features(word)
            s += self.create_inner_chars_features(word)

        s += "pt=" + pt + " "
        s += "ppt_pt=" + ppt + "_" + pt + " "
        s += "pw=" + pw + " "
        s += "ppw=" + ppw + " "
        s += "nw=" + nw + " "
        s += "nnw=" + nnw
        return s

    def _contains_number(self, s):
        digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        for d in digits:
            if d in s:
                return True
        return False

    def _create_prefix_features(self, word):
        s = ""
        for i in range(1, min(5, len(word))):
            pref = word[:i]
            s += "pref=" + pref + " "
        return s

    def _create_suffix_features(self, word):
        s = ""
        for i in range(1, min(5, len(word))):
            suffix = word[-i:]
            s += "suf=" + suffix + " "
        return s

    def create_inner_chars_features(self, word):
        s = ""
        if len(word) == 0:
            return s
        if self._contains_number(word):
            s += "number=True "
        if word[0].isupper():
            s += "upper=True "
        if "-" in word:
            s += "hyphen=True "
       # if len(s) > 0: s+=" "
        return s