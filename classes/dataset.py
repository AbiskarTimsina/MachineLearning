'''
Publisher: Abiskar Timsina
Project: Image classifier (Cat vs Non-Cat) using binary classification and deeplearning neural network.
This class: For retriving the h5 data set and converting into numpy array for ease.
'''

import numpy as np
import h5py

class datasets():
    def data(self):

        #for training sets
        train_data = h5py.File('datasets/train_catvnoncat.h5','r') #directory for the dataset
        train_set_x= np.array(train_data["train_set_x"][:])# the train_set_x is a feature within the h5 file
        train_set_y = np.array(train_data["train_set_y"][:])#label
        #for testing sets
        test_data= h5py.File('datasets/test_catvnoncat.h5','r')
        test_set_x = np.array(test_data["test_set_x"][:])#feature
        test_set_y = np.array(test_data["test_set_y"][:])#label

        return train_set_x,train_set_y, test_set_x, test_set_y
