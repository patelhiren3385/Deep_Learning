'''Binary Classification Problem Solved using :: MP Neuron'''
import pandas as pd
import sklearn.datasets
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
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
data = pd.DataFrame(cancer_data.data,columns=cancer_data.feature_names)
data['Labels'] = cancer_data.target #To add Lables columns to VIWEING DATA
xx = data.drop('Labels',axis=1) #This is Extraction of DATA in DATAFRAME TYPE
yy = data['Labels'] #This is Extraction of DATA in DATAFRAME TYPE
x_train,x_test,y_train,y_test = train_test_split(xx,yy,test_size=0.1,stratify=yy,random_state=1) #Stratify w.r.t. yy means we want to split the data in such a way that ratio of 0 and 1 reamain almost same in both test and train data: And Random state with int constant will make sure the data is split in same way each time we run the code means data is having same mean and other stat.

##############################################################################################
'''Binarising the Input for MP Neuron'''
##############################################################################################

#x_train_bin_mean_area = x_train['mean area'].map(lambda x: 0 if x < 1000 else 1) #Example how to split using MAP and Lambda
x_train_bin = x_train.apply(pd.cut,bins = 2,labels = [1,0]) #.apply will apply to all the columns in DATA SET (Type == Pandas DataFrame)
x_test_bin = x_test.apply(pd.cut,bins = 2,labels = [1,0]) #.apply will apply to all the columns in DATA SET (Type == Pandas DataFrame)
'''It is necessary to convert Pandas DataFrame into Numpy array for Applying Algo'''
x_train_bin = x_train_bin.values # .values will transform DataFrame --> Numpy array
x_test_bin = x_test_bin.values # .values will transform DataFrame --> Numpy array

##############################################################################################
'''Mp Neuron with Classes'''
##############################################################################################

class Mp_neuron:
    def __init__(self):
        self.b = None

    def model(self, x_1):
        return (np.sum(x_1) >= self.b)

    def learning_algo(self,x_2):
        y_1 = []
        for i in x_2:
            result = self.model(i)
            y_1.append(int(result))
        return (y_1)

    def evaluation(self,x_3,y_3):
        accuracy = {} #Accuracy = No. of Correct Prediction / Total number of Predictions
        for b in range(x_3.shape[1] + 1):
            self.b = b
            y_pred = self.learning_algo(x_3)
            accuracy[b] = accuracy_score(y_pred,y_3)
        key_max = max(accuracy,key = accuracy.get)
        print('Optimum value of b is:', key_max,'\nAccuracy of MP neuron model is:',accuracy[key_max])
mp_neu = Mp_neuron() #Initializing the Class
mp_neu.evaluation(x_train_bin,y_train)



