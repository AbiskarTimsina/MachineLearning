'''
Publisher: Abiskar Timsina
Project: Image recognition (Cat vs Non-Cat) using binary classification and deeplearning neural network.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
#different class / file is made so that the path for datasets isn't hard coded and easily available.
from classes.dataset import datasets

'''
Initialization before using any deeplearning algorithm.
Variables used:
training_x_o: features used for training imported from the datasets
testing_x_o:  features used for testing imported from the datasets
testing_y / training_y : labels defined
m_training: no of iterations in the given training dataset.
m_testing: no of iterations in the given testing dataset.

'''
#refer to dataset.py. Imports values form the dataset
#training_x_o repersents dataset before preprocessing
training_x_o,traing_y,testing_x_o,testg_y = datasets().data()
m_training= 209 #this is the no of examples, found by using .shape of the...
m_testing=50 #...given array (209,64,64,3) where 209 are no of examples 64==height==widht and 3 is RGB channel.
num_px = 64

#rearranging data for better implementation(All column-wise...)
training_x1 = training_x_o.reshape((64*64*3),209)
testing_x1= testing_x_o.reshape((64*64*3),50)
training_y = traing_y.reshape(1,209)
testing_y = testg_y.reshape(1,50)

#Now standardzing the pixel values i.e in picture datasets the flattened
#array is divided by the max pixel value i.e 255 for better implementation.
training_x_std = training_x1/255
testing_x_std= testing_x1/255

'''
Using logistic regression model we simply,
-define a function
-define a activator
-compute loss
'''
def sigmoid(z):
    s = (1/(1 + np.exp(-z)))
    return s

def initilization(nom):
    W = np.zeros((nom,1)) #nom~ no of parameters
    b = 0
    return W,b

'''
Forward propagation and backward propagation for determing the paramertes
and minimizing cost and determining the gradient decent.
'''
def propagation(w,b,X,Y):
    m = X.shape[1]
    #Forward propagation = from X to cost
    A = sigmoid(np.dot(w.T,X)+b) #a = y_cap = sigma(z)... z = wx +b
    J = (-1/m)*(np.sum(Y*np.log(A)+(1-Y)* np.log(1-A)))# error function. J is a notation for cost.
    #Backward propagation = form cost to w,b to find the gradient
    dw = (1/m) * np.dot(X,((A-Y).T))
    db = (1/m) * np.sum((A-Y))
    grad = {"dw": dw,"db": db} #as dictionary so that it is easy to find values of dw and db
    return grad,J
'''
This function optimizes w and b by running a gradient descent algorithm
Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
'''
def optimize(w,b,X,Y,num_iterations,learning_rate):
    costs =[]
    for i in range(num_iterations):
        #Cost and gradient calculation
        grads,cost = propagation(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]
        #update rule
        w = w - (learning_rate *dw)
        b = b - (learning_rate *db)
        #Record the costs
        if i % 100 == 0:
            costs.append(cost)

        parameters= {"w":w,"b":b}
        grads = {"bw":dw,"db":db}

        return parameters,grads,costs
'''
Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
'''
def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    #computing A for predicting the probability of it being 1 or a Cat
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    return Y_prediction
'''
"""
    Build the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()

    Returns:
    d -- dictionary containing information about the model.
    """
'''
def model(training_x_std,training_y,testing_x_std,testing_y,num_iterations=200,learning_rate=0.005):
    #Initialization of parametes with 0
    w,b = initilization(training_x_std.shape[0])
    #gradient descent
    parameters, grads,costs = optimize(w,b,training_x_std,training_y,num_iterations,learning_rate)
    #Retriving parameters w and b from dictionary params
    w = parameters["w"]
    b = parameters["b"]
    #Predicting test/train set examples
    Y_prediction_test = predict(w,b,testing_x_std)
    Y_prediction_train= predict(w,b,training_x_std)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(training_y - Y_prediction_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(testing_y - Y_prediction_test)) * 100))

    d = {"costs":costs,"Y_prediction_test":Y_prediction_test,"Y_prediction_train":Y_prediction_train,
    "w":w,"b":b,"learning_rate":learning_rate,"num_iterations":num_iterations}
    return d

d = model(training_x_std, training_y, testing_x_std, testing_y, num_iterations = 200, learning_rate = 0.005)
