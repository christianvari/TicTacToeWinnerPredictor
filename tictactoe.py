from tictactoe_functions import *

DATA_PATH = "./ticTacToeDataset.txt"

data = getData(DATA_PATH)
#Divido training e test set
train_set, test_set = doStratifiedShuffleSplit(data, data["Class"])
