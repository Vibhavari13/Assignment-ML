#!/usr/bin/env python
# coding: utf-8

# In[3]:


from random import seed
from random import randrange
from csv import reader
import numpy as np
import pandas as pd
from numpy.random import randn

import random
np.random.seed(101)


df = pd.DataFrame(randn(100,4),columns='W X Y Z'.split())
print(df)

ran=list(np.random.randint(2, size=100))
df['O']=ran

df=df.reset_index(drop=True)
print(df)
df.info()
#split
train=df.sample(frac=0.8,random_state=200) #random state is a seed value
test=df.drop(train.index)

print(train)




# In[4]:


l=[]
for i in train.values:
    l.append(list(i))
    
print(l)


# In[6]:


import seaborn as sns
# Split a dataset into k folds
import matplotlib.pyplot as plt

error_graph_data=[]
stor_weights=[]
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights
 
# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0


def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions) 






    
    
  
  
def find_acc_metrics(act, pred):
    tn,tp,fn,fp,= 0,0,0,0 
    #true negatve, true positive, false negative, false positive
    # 1=True, 0=False
    for i in range(len(act)): 
        if act[i] == 1 and pred[i]==1: 
            tp+=1
        elif act[i]==1 and pred[i]==0:
            fn+=1
        elif act[i]==0 and pred[i]==1:
            fp+=1
        else: 
            tn+=1
    #Accuracy, Confusion Matrix, Precision, Recall
    return [(tn+tp)/(tn+tp+fn+fp),str(tp)+'  '+str(fn)+'\n'+str(fp)+'  '+str(tn),tp/(tp+fp),tp/(tp+fn)]
 

# Estimate Perceptron weights using stochastic gradient descent


def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
        
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds) 
    #get folds
    scores = list() 
    #list of accuracies
    for i in range(len(folds)): 
        #for every fold
        train_set = list(folds) 
        #your training set will be all the folds minus the current fold
        train_set.remove(folds[i]) 
        #remove current fold 
        train_set = sum(train_set, []) 
        test_set = list() 
        #empty test set
        actual=list()
        for row in folds[i]:
            row_copy = list(row) 
            #get entire row in the fold
            actual.append(row_copy[-1]) 
            #append the true value
            row_copy[-1] = None 
            #set the to predict attribute to "None"
            test_set.append(row_copy) 
            #append it into the test set
        
        predicted = algorithm(train_set, test_set, *args) 
        #get predictions from the MLP algorithm
        metrics = find_acc_metrics(actual, predicted) 
        #calculate the metrics
        print('-------Fold',i+1,'------')
        print('*****Hyperparameters*****') #print the hyperparameters
        print('Cumulative Epochs: ',5000*(i+1)) #print results after nth epoch
        print('Learning rate: ',0.02,'\n') #alpha 
        print('METRICS')
        print('Accuracy: ',metrics[0]) 
        print('Confusion Matrix:\n'+metrics[1])
        print('Precision: ',metrics[2])
        print('Recall: ',metrics[3],'\n')
        scores.append(metrics[0]) #append into scores
    return scores

	
 
# Perceptron Algorithm With Stochastic Gradient Descent

 
# Test the Perceptron algorithm on the sonar dataset
seed(1)
# load and prepare data


l_rate = 0.5
n_epoch = 5
train_weights(l, l_rate, n_epoch)




n_folds = 3
l_rate = 0.01
n_epoch = 100
scores = evaluate_algorithm(l, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' %scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

    
for row in l:
    print(predict(row,weights))
    

    


    


# In[ ]:





# In[ ]:




