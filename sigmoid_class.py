import numpy as np
import pandas as pd
import sklearn.datasets
from matplotlib import pyplot as plt
from pip._vendor.colorama import initialise
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
###########################################################################################################
# Data - Real Inputs
# Task - Classification and Regression
# Model - sigmoid function - y^ = 1/(1 + exp^(-w*x + b))
# Loss -  Squared Error Loss
# Learning Algorithm - Gradient Descent
#                      w = w - lr*dw and b = b - lr*db :: dw = (y^ - y)*y^*(1-y^)*x :: db = (y^ - y)*y^*(1-y^)
# Evaluation - Accuracy = No. of correct prediction/Total No. of Prediction

###########################################################################################################
# Loading Data
###########################################################################################################

cancer_data = sklearn.datasets.load_breast_cancer() # Regression data
x = cancer_data.data
y = cancer_data.target

###########################################################################################################
# Processing Data
###########################################################################################################

data = pd.DataFrame(x,columns=cancer_data.feature_names)
data['Labels'] = cancer_data.target
xx = data.drop('Labels',axis = 1)
yy = data['Labels']
# Standardization of data so that large data is scaled up to Zero mean and Unit variance
scalar = StandardScaler()
std_prop = scalar.fit(xx.values)
std_xx = scalar.transform(xx)
std_yy = yy

x_train,x_test,y_train,y_test = train_test_split(std_xx,std_yy,test_size=0.1,random_state=1,stratify=std_yy)

###########################################################################################################
# Sigmoid Class
###########################################################################################################

class Sigmoid:


    def __init__(self):
        self.w = None
        self.b = None

    def perceptron(self,x):
        return np.dot(x, self.w) + self.b

    def sig(self,x):
        return 1 / (1 + np.exp(self.perceptron(-x)))

    def grad_w(self,x,y):
        y_pred = self.sig(x)
        return (y_pred-y)*y_pred*(1-y_pred)*x

    def grad_b(self,x,y):
        y_pred = self.sig(x)
        return (y_pred - y)*y_pred*(1 - y_pred)

    def DataStandardization(self, X_train = np.zeros((10,2)), Y_train = np.zeros((10,2)), X_test = np.zeros((10,2)), Y_test = np.zeros((10,2))):
        scalar = StandardScaler()
        minmax_scalar = MinMaxScaler()
        X_scaled_train = scalar.fit_transform(X_train)
        Y_scaled_train = minmax_scalar.fit_transform(Y_train.reshape(-1,1))
        X_scaled_test = scalar.fit_transform(X_test)
        Y_scaled_test = minmax_scalar.fit_transform(Y_test.reshape(-1,1))
        return X_scaled_train, X_scaled_test, Y_scaled_train, Y_scaled_test

    def fit(self,X,Y,epochs = 1,initialise = True , display_loss = True,lr = 0.01):
        if initialise:
            self.w = np.ones(np.size(X,1))
            self.b = 0
        if display_loss:
            loss = {}
        for i in range(epochs):
            dw = 0
            db = 0
            for x,y in zip(X,Y):
                dw = dw + self.grad_w(x,y)
                db = db + self.grad_b(x,y)
            self.w = self.w - lr*dw
            self.b = self.b - lr*db
            if display_loss:
                Y_pred = self.sig(X)
                loss[i] = mean_squared_error(Y,Y_pred)


        if display_loss:
            plt.plot(np.array(list(loss.values())).astype(float))
            plt.xlabel('No. of epochs')
            plt.ylabel('Loss')
            plt.grid()
            plt.show()
            plt.savefig('loss')



sigmo = Sigmoid()
res1 = sigmo.fit(x_train,y_train,100)















