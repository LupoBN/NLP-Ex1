import numpy as np
from Helpers import *

SUFFIXES = ["ing", "ed", "s", "er", "est", "dom", "ism", "ist", "al", "ity", "ment", "ness", "tion",
            "ship", "ate", "en", "ify", "fy", "sion", "ize", "able", "ful", "ish", "less", "ive"]
PREFIXES = ["un", "de", "re", "in", "anti", "auto", "Auto", "Anti", "Un", "De", "Re", "In", "im", "Im", "Pre", "pre",
            "extra", "Extra", "over", "Over"]


def count_labels(labels, order):
    labels_count = dict()
    labels_set = set()
    for i in range(1, order + 1):
        for j in range(0, len(labels) - i + 1):
            labels_set.add(labels[j])
            order_gram = " ".join(reversed(labels[j:i + j]))
            if order_gram not in labels_count:
                labels_count[order_gram] = 1
            else:
                labels_count[order_gram] += 1
    return labels_count, labels_set


def label_word_pairs(words_and_labels):
    pairs_count = dict()
    word_count = dict()
    for word, label in words_and_labels:
        pair = word + " " + label
        if pair not in pairs_count:
            pairs_count[pair] = 1
        else:
            pairs_count[pair] += 1
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1

    return pairs_count, word_count


def calculate_UNK_probs(words_and_labels):
    first_half = words_and_labels[:len(words_and_labels) / 2]
    first_haft_words = set()
    second_half = words_and_labels[len(words_and_labels) / 2:]
    unk_dict = dict()
    for word, label in first_half:
        first_haft_words.add(word)
    for word, label in second_half:
        if word not in first_haft_words:
            word_pair = "UNK " + label
            if word_pair in unk_dict:
                unk_dict[word_pair] += 1
            else:
                unk_dict[word_pair] = 1
    return unk_dict