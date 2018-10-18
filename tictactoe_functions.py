import pandas as pd
from os import path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder


#Utility per caricare il csv
def getData (data_path):
	data_path = path.join(data_path)
	return pd.read_csv(data_path)

def doStratifiedShuffleSplit(data ,stratifying_attribute):
	split = StratifiedShuffleSplit(test_size = 0.2, random_state=42)
	for train_index, test_index in split.split(data, stratifying_attribute):
		strat_train_set = data.loc[train_index]
		strat_test_set = data.loc[test_index]

	return strat_train_set, strat_test_set
