#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Dec 14, 2017

@author: timekeeper
"""
import gensim
import re
from itertools import groupby
import numpy as np
from sklearn.model_selection import train_test_split


class DataTransform(object):
    def __init__(self):
        """
        Constructor
        """
        self.__rx = re.compile(u'[\d]+|/+|[_+#=%><№$€…«»|/*{}\[\]‘]|[(\d)]|[\[\]]')

    def __del__(self):
        """
        Destructor
        """

    def punctuation_replacement(self, inline):
        line = gensim.utils.to_unicode(inline).lower()
        line = self.__rx.sub(u' ', line)
        return line.replace(u'\n', u' ').replace(u'\\', u' ').split()

    @staticmethod
    def array2file(fname, corpus):
        out_file = open(fname, "w")
        for s in corpus:
            out_file.write(" ".join(s).encode('utf-8'))
            out_file.write("\n")
        out_file.close()

    def file2vec(self, fname, start=0, end=0):
        sentences = []

        sentences = []
        assert (start >= 0)
        assert (end >= 0)
        if end == 0:
            with open(fname, "r") as in_file:
                for inline in in_file.readlines()[start:]:
                    l = self.punctuation_replacement(inline)
                    if len(l) > 0:
                        sentences.append([el for el, _ in groupby(l)])
        else:
            with open(fname, "r") as in_file:
                for inline in in_file.readlines()[start:end]:
                    l = self.punctuation_replacement(inline)
                    if len(l) > 0:
                        sentences.append([el for el, _ in groupby(l)])
        return sentences

    def str2vec(self, inline):
        sentences = []
        l = self.punctuation_replacement(inline)
        if len(l) > 0:
            sentences.append([el for el, _ in groupby(l)])
        return sentences

    @staticmethod
    def labelize_reviews(reviews, class_label, label_type):
        LabeledSentence = gensim.models.doc2vec.TaggedDocument
        labelized = []
        l_type = ""
        if len(class_label) > 0:
            for i, v in enumerate(reviews):
                if class_label[i] == 0:
                    l_type = label_type + "_NEGATIVE"
                elif class_label[i] == 1:
                    l_type = label_type + "_POSITIVE"
                label = '%s_%s' % (l_type, i)
                labelized.append(LabeledSentence(v, [label]))
        else:
            for i, v in enumerate(reviews):
                label = '%s_%s' % (label_type, i)
                labelized.append(LabeledSentence(v, [label]))
        return labelized

    @staticmethod
    def get_vecs(model, corpus, r_size):
        vecs = [np.array(model.docvecs[z.tags]).reshape((1, r_size)) for z in corpus]
        return np.concatenate(vecs)

    @staticmethod
    # Get vectors from our models
    def get_vecs_by_words(model, corpus, r_size):
        vecs = [np.array(model.infer_vector(z.words)).reshape((1, r_size)) for z in corpus]
        return np.concatenate(vecs)

    @staticmethod
    def split_learn_set(sentences_positive, sentences_negative, test_size=0.2):
        y = np.concatenate(
            (np.ones(len(sentences_positive), dtype=np.int), np.zeros(len(sentences_negative), dtype=np.int)))
        x_train, x_test, y_train, y_test = train_test_split(np.concatenate((sentences_positive, sentences_negative)), y,
                                                            test_size=test_size)
        return x_train, x_test, y_train, y_test

    def create_learn_sets(self, sent_positive, sent_negative, sent_unsup, t_size=0.2):
        x_train, x_test, y_train, y_test = self.split_learn_set(sentences_positive=sent_positive,
                                                                sentences_negative=sent_negative,
                                                                test_size=t_size)
        x_train = self.labelize_reviews(x_train, y_train, "TRAIN")
        x_test = self.labelize_reviews(x_test, y_test, "TEST")
        x_unsup = self.labelize_reviews(sent_unsup, [], "UNSUP")
        return dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, x_unsup=x_unsup)
