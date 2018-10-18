from tictactoe_functions import *
from sklearn.metrics import confusion_matrix


DATA_PATH = "./ticTacToeDataset.txt"

data = getData(DATA_PATH)
doEncodeTextData(data)

# Divido training e test set
train_set, test_set, train_label_set, test_label_set = doStratifiedShuffleSplit(data, "Class", "Class")

train_label_set = (train_label_set == 1)
test_label_set = (test_label_set == 1)

classificatore = createClassificator(train_set,train_label_set)
predizioni = doPredictions(train_set, classificatore)

print(confusion_matrix(predizioni, train_label_set))
