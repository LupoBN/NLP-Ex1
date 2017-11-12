from DataManager import *
import sys
if __name__ == '__main__':
    words_and_labels = read_file(sys.argv[1], parse_pos_reading)
    words = [data[0] for data in words_and_labels]
    labels = [data[1] for data in words_and_labels]

    labels_count, labels_set = count_labels(labels, 3)
    write_file(sys.argv[2], labels_count, parse_pos_writing)

    pairs_count, word_count = label_word_pairs(words, labels)
    write_file(sys.argv[3], pairs_count, parse_pos_writing)
    possible_labels = parse_possible_labels(words_and_labels)
    write_file("data/possible_labels.txt", possible_labels, parse_possible_labels_writing)

