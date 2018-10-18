import pandas as pd
from os import path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
import numpy as np


def getData(data_path):
    data_path = path.join(data_path)
    return pd.read_csv(data_path)


def doStratifiedShuffleSplit(data, stratifying_attribute, label_attribute):
    split = StratifiedShuffleSplit(test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data[stratifying_attribute]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    train_label = strat_train_set[label_attribute]
    test_label = strat_test_set[label_attribute]
    strat_train_set = strat_train_set.drop(labels=label_attribute, axis=1)
    strat_test_set = strat_test_set.drop(labels=label_attribute, axis=1)
    print("Train: ", strat_train_set.shape, "label : ", train_label.shape, " Test: ", strat_test_set.shape, " label: ", test_label.shape)
    return strat_train_set, strat_test_set, train_label, test_label


def doEncodeTextData(data):
    for c in data.columns:

        encoder = LabelEncoder()
        data[c] = encoder.fit_transform(data[c])


def createClassificator(train, train_labels):
    classificatore = SGDClassifier(random_state=42)
    classificatore.fit(train, train_labels)
    return classificatore


def doPredictions(data, classificatore):
    array = np.array(data)
    return classificatore.predict(array)
