from tictactoe_functions import *


DATA_PATH = "./ticTacToeDataset.txt"

data = getData(DATA_PATH)
data = doEncodeTextData(data)

# Divido training e test set
train_set, test_set, train_label_set, test_label_set = doStratifiedShuffleSplit(data, "Class", "Class")

train_label_set = (train_label_set == 1)
test_label_set = (test_label_set == 1)

classificatore = createClassificator(train_set, train_label_set)
predizioni_train = doPredictions(train_set, classificatore)

print("\n########################################\nTrain Metrics:")
printMetrics(train_label_set, predizioni_train)

predizioni_test = doPredictions(test_set, classificatore)

print("\n########################################\nTest Metrics:")
printMetrics(test_label_set, predizioni_test)
