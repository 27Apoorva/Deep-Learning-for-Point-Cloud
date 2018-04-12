#understanding LSTM
from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse
import pdb
# data_path = "/home/arpit/Deep_Learning_Project/LSTM/simple-examples/data"
# data_path = "/home/appu/DeepLearning/Project/LSTM/simple-examples/data"
data_path = "/home/appu/DeepLearning/Project/Deep-Learning-for-Point-Cloud/Point Cloud/2011_09_26/2011_09_26_drive_0001_extract/velodyne_points/data"
parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()
if args.data_path:
    data_path = args.data_path

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        pdb.set_trace()
        return f.read().split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]



# def load_data():
#     # get the data paths
#     train_path = os.path.join(data_path, "ptb.train.txt")
#     valid_path = os.path.join(data_path, "ptb.valid.txt")
#     test_path = os.path.join(data_path, "ptb.test.txt")
#
#     # build the complete vocabulary, then convert text data to list of integers
#     word_to_id = build_vocab(train_path)
#     train_data = file_to_word_ids(train_path, word_to_id)
#     valid_data = file_to_word_ids(valid_path, word_to_id)
#     test_data = file_to_word_ids(test_path, word_to_id)
#     vocabulary = len(word_to_id)
#     reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
#     print("train data")
#     print(train_data[:3])
#     pdb.set_trace()
#     #print(word_to_id)
#     print(vocabulary)
#     print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
#     return train_data, valid_data, test_data, vocabulary, reversed_dictionary
#
# train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()

def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "0000000000.txt")
    train_data = read_words(train_path)
    print("train data")
    print(train_data[:3])
    pdb.set_trace()
    return train_data

train_data = load_data()
