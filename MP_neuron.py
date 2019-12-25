'''Binary Classification Problem Solved using :: MP Neuron'''
import pandas as pd
import sklearn.datasets
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
##############################################################################################
'''Loading DATA'''
cancer_data = sklearn.datasets.load_breast_cancer() #Binary Classification Problem DATA
x = cancer_data.data # .data will give FEATURES
y = cancer_data.target # .target will give LABELS
##############################################################################################
'''Data Processing'''
#pd.set_option('display.max_columns',None) #Viewing Options All the Columns
data = pd.DataFrame(cancer_data.data,columns=cancer_data.feature_names)
data['Labels'] = cancer_data.target #To add Lables columns to VIWEING DATA
xx = data.drop('Labels',axis=1) #This is Extraction of DATA in DATAFRAME TYPE
yy = data['Labels'] #This is Extraction of DATA in DATAFRAME TYPE
x_train,x_test,y_train,y_test = train_test_split(xx,yy,test_size=0.1,stratify=yy,random_state=1) #Stratify w.r.t. yy means we want to split the data in such a way that ratio of 0 and 1 reamain almost same in both test and train data: And Random state with int constant will make sure the data is split in same way each time we run the code means data is having same mean and other stat.
##############################################################################################
'''Binarising the Input for MP Neuron'''
#x_train_bin_mean_area = x_train['mean area'].map(lambda x: 0 if x < 1000 else 1) #Example how to split using MAP and Lambda
x_train_bin = x_train.apply(pd.cut,bins = 2,labels = [1,0]) #.apply will apply to all the columns in DATA SET (Type == Pandas DataFrame)
x_test_bin = x_test.apply(pd.cut,bins = 2,labels = [1,0]) #.apply will apply to all the columns in DATA SET (Type == Pandas DataFrame)
'''It is necessary to convert Pandas DataFrame into Numpy array for Applying Algo'''
x_train_bin = x_train_bin.values # .values will transform DataFrame --> Numpy array
x_test_bin = x_test_bin.values # .values will transform DataFrame --> Numpy array
##############################################################################################
'''Learning Algorithm'''
y_pred_train = []
accu = []
for b in range(0,x_train_bin.shape[1]): #Since maximum Possible value of b will be No of Features
    y_correct = 0 #It is put here because we want to calculate accuracy for each value of 'b'
    for i, j in zip(x_train_bin,y_train):
        y_pred = (np.sum(i)>=b) #This will giver boolean output
        y_pred_train.append(int(y_pred)) #Putting int(boolean) will convert boolean to '1' and '0'
        y_correct += (j == y_pred) #If condition met it will add value to y_correct
    accu.append(y_correct)
##############################################################################################
print('Optimum value of b is:',accu.index(max(accu)),'\nAccuracy of MP neuron model is:',max(accu)/x_train_bin.shape[0])



