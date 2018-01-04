#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""

Created on Nov 07, 2016

@author: timekeeper
"""

import logging.config
from sys import exit
from src import config
logging.config.dictConfig(config.LOGGING_CONFIG)
try:
    import gensim
    import numpy as np
    from src.data_transform import DataTransform
    from src.decision_model import DecisionModel
    import random as rnd
    from sklearn.metrics import roc_curve, auc, classification_report
    import matplotlib.pyplot as plt
except Exception as exc:
    logging.error(exc)
    logging.info("Please use 'pip3 install <module_name>' to install module")
    exit()

SIZE = 400
EPOCHS = 20
WORKERS = 2
LOAD_MODEL = True

if __name__ == '__main__':
    logging.info("START")
    dt = DataTransform()

    # Load data
    positive_data_file = config.DATA_PATH + "pos.data"
    negative_data_file = config.DATA_PATH + "neg.data"
    unsup_data_file = config.DATA_PATH + "unsup.data"

    model_dm_file = config.MODEL_PATH + "model_dm_{}_{}_{}.d2v".format(SIZE, EPOCHS, WORKERS)
    model_dbow_file = config.MODEL_PATH + "model_dbow_{}_{}_{}.d2v".format(SIZE, EPOCHS, WORKERS)

    sentences_positive = dt.file2vec(fname=positive_data_file, start=0, end=0)
    sentences_negative = dt.file2vec(fname=negative_data_file, start=0, end=0)
    sentences_unsup = dt.file2vec(fname=unsup_data_file, start=0, end=0)

    # Create learning sets
    learn_set = dt.create_learn_sets(sent_positive=sentences_positive,
                                     sent_negative=sentences_negative,
                                     sent_unsup=sentences_unsup,
                                     t_size=0.2)

    model_dm = gensim.models.Doc2Vec(min_count=1, max_vocab_size=2e5, iter=10, window=10, size=SIZE, sample=1e-3,
                                     negative=5, workers=WORKERS)
    model_dbow = gensim.models.Doc2Vec(min_count=1, max_vocab_size=2e5, iter=10, window=10, size=SIZE, sample=1e-3,
                                       negative=5, dm=0, workers=WORKERS)

    # If we have learnd models, we can load it
    if LOAD_MODEL:
        model_dm = gensim.models.Doc2Vec.load(model_dm_file)
        model_dbow = gensim.models.Doc2Vec.load(model_dbow_file)

    all_docs = learn_set['x_train'] + learn_set['x_test'] + learn_set['x_unsup']
    train_docs = learn_set['x_train'] + learn_set['x_unsup']

    # Learn new models
    if not LOAD_MODEL:
        # Build vocabulary over all documents
        model_dm.build_vocab(all_docs, progress_per=1000, update=False)
        model_dbow.build_vocab(all_docs, progress_per=1000, update=False)

        # Fit the models on the train data
        logging.info("     Start fitting...")
        for epoch in range(EPOCHS):
            logging.info("         EPOCH {} begins".format(epoch, ))
            rnd.shuffle(train_docs)
            model_dm.train(train_docs, total_examples=model_dm.corpus_count, epochs=model_dm.iter)
            model_dbow.train(train_docs, total_examples=model_dbow.corpus_count, epochs=model_dbow.iter)
            logging.info("         EPOCH {} ends".format(epoch))
        logging.info("     Finish fitting")

        model_dm.save(model_dm_file)
        model_dbow.save(model_dbow_file)

    # Create train and test vectos from our models
    train_vecs_dm = dt.get_vecs_by_words(model_dm, learn_set['x_train'] + learn_set['x_test'], SIZE)
    train_vecs_dbow = dt.get_vecs_by_words(model_dbow, learn_set['x_train'] + learn_set['x_test'], SIZE)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

    test_vecs_dm = dt.get_vecs_by_words(model_dm, learn_set['x_test'], SIZE)
    test_vecs_dbow = dt.get_vecs_by_words(model_dbow, learn_set['x_test'], SIZE)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
    y = np.hstack((learn_set['y_train'], learn_set['y_test']))

    #
    decision_model = DecisionModel(train_vecs, y, seed=9, test_size=0.25)
    model, accuracy, class_report, conf_matrix, imps = decision_model.xgb()
    logging.info('     Test Accuracy (XGBClassifier): %.2f ' % accuracy)
    xgb_predictions = model.predict_proba(test_vecs)[:, 1]
    fpr, tpr, _ = roc_curve(learn_set['y_test'], xgb_predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', label='XGBClassifier: area = %.2f' % roc_auc)
    xgb_predictions = [round(value) for value in xgb_predictions]
    report = classification_report(learn_set['y_test'], xgb_predictions, target_names=['Neg', 'Pos'])
    print(report)

    model, accuracy, class_report, conf_matrix = decision_model.logistic_regression()
    logging.info('     Test Accuracy (LogisticRegression): %.2f ' % accuracy)
    lr_predictions = model.predict_proba(test_vecs)[:, 1]
    fpr, tpr, _ = roc_curve(learn_set['y_test'], lr_predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='green', label='LogisticRegression: area = %.2f' % roc_auc)
    lr_predictions = [round(value) for value in lr_predictions]
    report = classification_report(learn_set['y_test'], lr_predictions, target_names=['Neg', 'Pos'])
    print(report)

    model, accuracy, class_report, conf_matrix = decision_model.svc()
    logging.info('     Test Accuracy (SVC): %.2f ' % accuracy)
    svc_predictions = model.predict_proba(test_vecs)[:, 1]
    fpr, tpr, _ = roc_curve(learn_set['y_test'], svc_predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='brown', label='SVC: area = %.2f' % roc_auc)
    svc_predictions = [round(value) for value in svc_predictions]
    report = classification_report(learn_set['y_test'], svc_predictions, target_names=['Neg', 'Pos'])
    print(report)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.show()
