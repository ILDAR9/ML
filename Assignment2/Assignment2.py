# -*- coding: utf-8 -*-
__author__ = "ildar"
"""
# Assignment 2, let's consider the lab problem in week 4 again and try to use the same feature sets
# for perceptron and SVM. Wring a report on the comparison of the three methods discussing their
# performance of convergence (some tolerance may apply), misclassified emails in cases of various
# testing example sizes as in the second question, you may also discuss some other properties.
"""
import glob
import os
from collections import OrderedDict

CLASS_HUB = 0
CLASS_SPAM = 1
folders_test = ['nonspam-test', 'spam-test']
folders_train = ['nonspam-train', 'spam-train']
train_bound = 700
test_bound = 26
folder_output = 'output_test_' + str(test_bound)
output_dict = os.path.join(folder_output, 'dictionary.txt')
output_train_feature = os.path.join(folder_output, 'train-features.txt')
output_train_labels = os.path.join(folder_output, 'train-labels.txt')
output_train_files = os.path.join(folder_output, 'train-files.txt')
output_test_feature = os.path.join(folder_output, 'test-features.txt')
output_test_labels = os.path.join(folder_output, 'test-labels.txt')
output_test_files = os.path.join(folder_output, 'test-files.txt')

# list files
f_list_files = lambda folder: sorted((f for f in glob.glob(os.path.join(folder, '*.txt'))))
# count occupancies
f_occurrence = lambda tokens: dict(((word, tokens.count(word)) for word in set(tokens)))
# sort dict
f_sort_dict = lambda d: OrderedDict((k, d[k]) for k in sorted(d, key=d.get, reverse=True))
# read files
def get_tokens(file):
    tokens = []
    f = open(file, 'r')
    for line in f:
        tokens.extend(filter(lambda word: len(word) > 1, str(line).split()))
    f.close()
    return tokens


#########################################################################
# multinomial Naive Bayes model feature creator for Matlab/Octave process
#########################################################################
def create_bag_words(track_bag_dict=None):
    bag_words = {}
    for folder in folders_test:
        process_folder(folder, bag_words)
    for folder in folders_train:
        process_folder(folder, bag_words, track_bag_dict)
    return f_sort_dict(bag_words)


def process_folder(folder, bag_words, track_dict=None):
    list_files = f_list_files(folder)
    for file in (list_files[:test_bound] if track_dict is None else list_files[: train_bound]):
        dict_f = f_occurrence(get_tokens(file))
        if track_dict is not None: track_dict.append(dict_f)
        for k, v in dict_f.items():
            bag_words[k] = bag_words[k] + v if k in bag_words else v


def persist_features(bag_words_c, bag_words_d_list, output_features):
    f = open(output_features, 'w')
    doc_i = 0
    list_index = list(bag_words_c.keys())
    print("================", len(bag_words_d_list))
    for bag_words_d in bag_words_d_list:
        doc_i += 1
        for word in sorted(bag_words_d, key=lambda x: list_index.index(x)):
            f.write('{}, {}, {}\n'.format(doc_i, list_index.index(word) + 1, bag_words_d[word]))
    f.close()


def persist_dict(bag_words_c):
    f = open(output_dict, 'w')
    for k, v in bag_words_c.items():
        f.write('{}:{}\n'.format(k, v))
    f.close()


def persist_label(folder, output_label_f, output_files_f):
    f_label = open(output_label_f, 'w')
    f_file = open(output_files_f, 'w')
    for type in [CLASS_HUB, CLASS_SPAM]:
        list_files = f_list_files(folder[type])
        for file_name in (list_files[:test_bound] if 'test-' in output_label_f else list_files[:train_bound]):
            f_label.write('{}\n'.format(type))
            f_file.write('{}\n'.format(file_name))
    f_label.close()
    f_file.close()


def generate_features_train_with_dict():
    if not os.path.exists(folder_output): os.makedirs(folder_output)
    bag_words_d_list = []
    bag_words_c = create_bag_words(bag_words_d_list)
    i = 0
    for x in bag_words_c:
        print(x, bag_words_c[x])
        i += 1
        if i > 5: break

    persist_features(bag_words_c, bag_words_d_list, output_train_feature)
    persist_dict(bag_words_c)
    persist_label(folders_train, output_train_labels, output_train_files)


def read_dictionary():
    f = open(output_dict, 'r')
    bag_words_c = OrderedDict(tuple(str(row).strip().split(':')) for row in f)
    f.close()
    return bag_words_c


def generate_features_test():
    bag_words_c = read_dictionary()
    i = 0
    for x in bag_words_c:
        print(x, bag_words_c[x])
        i += 1
        if i > 5: break
    print('------------')
    persist_label(folders_test, output_test_labels, output_test_files)
    bag_words_d_list = [f_occurrence(get_tokens(file)) for file in
                        f_list_files(folders_test[CLASS_HUB])[:test_bound] + f_list_files(folders_test[CLASS_SPAM])[
                                                                             :test_bound]]
    persist_features(bag_words_c, bag_words_d_list, output_test_feature)


generate_features_train_with_dict()
generate_features_test()
