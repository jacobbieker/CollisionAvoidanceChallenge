from keras.layers import LSTM, Dense, Input, Conv1D, Flatten, TimeDistributed, MaxPool1D
from keras.models import Sequential
import numpy as np


def read_data(filepath):
    data = np.genfromtxt(filepath, delimiter=",", names=True)
    # Now group by ID number into variable length 1D timeseries
    combined_data = []
    high_data = []
    low_data = []
    current_num = 0
    tmp = []
    tmp_low = []
    tmp_high = []
    for element in data:
        #element = np.delete(element, 2, 1)
        if int(element[0]) == current_num:
            if element[1] >= 2:
                tmp.append(element)
                if element[2] >= -6:
                    tmp_low.append(element)
                else:
                    tmp_high.append(element)
        else:
            combined_data.append(np.asarray(tmp))
            if len(tmp_high) > 0:
                high_data.append(np.asarray(tmp_high))
            if len(tmp_low) > 0:
                low_data.append(np.asarray(tmp_low))
            tmp = []
            tmp_low = []
            tmp_high = []
            current_num += 1
    combined_data = np.asarray(combined_data)
    high_data = np.asarray(high_data)
    low_data = np.asarray(low_data)
    return combined_data, high_data, low_data


train_data, train_low, train_high = read_data("train_data.csv")
print(train_data)
print(train_data.shape)
print(train_high.shape)
print(train_low.shape)


def create_model(input_shape=(32, 15, 103)):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(64, input_shape=input_shape)))
    model.add(TimeDistributed(MaxPool1D))
    model.add(TimeDistributed(Conv1D(64)))
    model.add(TimeDistributed(MaxPool1D))
    model.add(TimeDistributed(Conv1D(128)))
    model.add(TimeDistributed(MaxPool1D))
    model.add(TimeDistributed(Conv1D(128)))
    model.add(TimeDistributed(MaxPool1D))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(1))

    return model
