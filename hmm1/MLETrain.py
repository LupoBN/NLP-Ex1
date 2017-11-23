import sys

sys.path.insert(0, '../Extras')
from DataManager import *
import sys

if __name__ == '__main__':
    words_and_labels = read_file(sys.argv[1], parse_pos_reading)
    unk_count = calculate_UNK_probs(words_and_labels)
    labels = [data[1] for data in words_and_labels]
    labels_count, labels_set = count_labels(labels, 3)
    write_file(sys.argv[2], labels_count, parse_pos_writing)
    pairs_count, word_count = label_word_pairs(words_and_labels)
    pairs_count.update(unk_count)
    write_file(sys.argv[3], pairs_count, parse_pos_writing)
