##############################################################################################
'''Binary Classification Problem Solved using :: Perceptron'''
##############################################################################################

import pandas as pd
import sklearn.datasets
import numpy as np
import torch
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##############################################################################################
'''Loading DATA'''
##############################################################################################

cancer_data = sklearn.datasets.load_breast_cancer() #Binary Classification Problem DATA
x = cancer_data.data # .data will give FEATURES
y = cancer_data.target # .target will give LABELS

##############################################################################################
'''Data Processing'''
##############################################################################################

#pd.set_option('display.max_columns',None) #Viewing Options All the Columns
data = pd.DataFrame(x,columns=cancer_data.feature_names)
data['Labels'] = cancer_data.target #To add Lables columns to VIWEING DATA
xx = data.drop('Labels',axis=1) #This is Extraction of DATA in DATAFRAME TYPE
yy = data['Labels'] #This is Extraction of DATA in DATAFRAME TYPE
x_train,x_test,y_train,y_test = train_test_split(xx,yy,test_size=0.1,stratify=yy,random_state=1) #Stratify w.r.t. yy means we want to split the data in such a way that ratio of 0 and 1 reamain almost same in both test and train data: And Random state with int constant will make sure the data is split in same way each time we run the code means data is having same mean and other stat.

##############################################################################################
'''Perceptron with Classes'''
##############################################################################################

#############################################
# Model - y_pred = 1 if sum_i = w_i*x_i >= b MODEL - (1).  Model (2). Predict
#                = 0 Otherwise
# Learning Algo - Update rule
#           if y == 1 and y_pred == 1
#               w = w + x
#           elif y == 1 and y_pred == 0
#               w = w - x
class Perceptron:

    def __init__(self):
        self.b = None
        self.w = None

    def model(self,x):
        return 1 if (np.dot(self.w,x) >= self.b) else 0

    def predict(self,x_mat):
        y_pred = []
        for x in x_mat:
            result = self.model(x)
            y_pred.append(result)
        return np.array(y_pred)

    def fit(self,X,Y,epochs = 1,lr = 1):
        self.w = np.ones(np.size(X, 1))
        self.b = 1
        accu = {}
        accuracy = []
        max_accuracy = 0
        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0 :
                    self.w = self.w + lr*x
                    self.b = self.b - lr*1
                elif y == 0 and y_pred == 1 :
                    self.w = self.w - lr*x
                    self.b = self.b + lr * 1
            accu[i] =  accuracy_score(self.predict(X),Y)
            accuracy.append(accu[i])
            if (accu[i] > max_accuracy):
                max_accuracy = accu[i]
                check_point_w = self.w
                check_point_b = self.b

        self.w = check_point_w
        self.b = check_point_b
        key_max = max(accu, key=accu.get)
        print('Maximum Accuracy on Train data is: ',max_accuracy)
        print('Optimum Value of b is: ',self.b)
        plt.plot(accuracy)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy Score')
        plt.title('Perceptron - Binary Classification')
        plt.grid(True)
        plt.ylim([0,1])
        plt.savefig('Perceptron - Binary Classification.pdf', bbox_inches='tight')
        plt.savefig('Perceptron - Binary Classification.png', bbox_inches='tight')
        plt.savefig('Perceptron - Binary Classification.eps', bbox_inches='tight')
        plt.show()

perceptron_cancer = Perceptron()
result_train = perceptron_cancer.fit(x_train.values,y_train,100,0.01)
Y_pred_test = perceptron_cancer.predict(x_test.values)
print('Maximum Accuracy on Test data is: ',accuracy_score(Y_pred_test, y_test))






