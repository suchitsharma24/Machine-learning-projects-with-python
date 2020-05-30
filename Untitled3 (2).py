#!/usr/bin/env python
# coding: utf-8

# In[46]:


get_ipython().system('pip install mlxtend')


# In[47]:


from mlxtend.data import loadlocal_mnist
import os
os.listdir('D:\MNIST')


# In[48]:


train_x, train_y = loadlocal_mnist(
        images_path=r'D:\MNIST\train-images.idx3-ubyte', 
        labels_path=r'D:\MNIST\train-labels.idx1-ubyte')
test_x, test_y =loadlocal_mnist(
        images_path=r'D:\MNIST\t10k-images.idx3-ubyte', 
        labels_path=r'D:\MNIST\t10k-labels.idx1-ubyte')




# In[49]:


train_x.shape


# In[50]:


train_y.shape


# In[194]:


# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold


# In[263]:


from sklearn.neighbors import KNeighborsClassifier
loocv = model_selection.LeaveOneOut()
def KNN_leaveOneOut_score(train_x, train_y, test_x, test_y, K_max=7):
    score_train=list()
    score_test=list()
    #K_max=7 #20

    for K in range(1,K_max+1):
       
        
        #training score

        model_loocv = KNeighborsClassifier(n_neighbors=K)
        model_loocv.fit(train_x, train_y)
        results_loocv = model_selection.cross_val_score(model_loocv, train_x[:1000], train_y[:1000], cv=loocv)  #change 1000 to more to improve scoring
        score_train.append(results_loocv)
        
        #testing score
        results_loocv_test = model_selection.cross_val_score(model_loocv, test_x[:1000], test_y[:1000], cv=loocv)
        score_test.append(results_loocv_test)
    
    
    mean_score_train=list()
    for i in score_train:
        mean_score_train.append(np.mean(i))

    mean_score_test=list()
    for i in score_test:
        mean_score_test.append(np.mean(i))
    print("Training score",mean_score_train, type(mean_score_train))
    print("Testing score",mean_score_test, type(mean_score_test) )
#    return mean_score_train, mean_score_test   #Returns training score and testing score
    return mean_score_train, mean_score_test



    


# In[264]:



KNN_leaveOneOut_score(train_x, train_y, test_x, test_y, K_max=7)


# In[ ]:





# In[265]:


import time




start_time = time.time()
K_max=7

score_train, score_test = KNN_leaveOneOut_score(train_x, train_y, test_x, test_y, K_max=7)


import matplotlib.pyplot as plt


plt.plot(range(1,K_max+1), score_train, label='Train')
plt.plot(range(1,K_max+1), score_test, label='Test')
plt.legend()
plt.xlabel('K values')
plt.ylabel('KNN LeaveOneOutscore score')
plt.show()

print("Time of execution: ",time.time()-start_time)








# In[ ]:







# In[ ]:





# # PCA (Principle Component Analysis)

# In[266]:


run_time=dict() #for storing code running time for different PCA n values


# In[275]:


#Dimensionality reduction to reduce number of variables for representing our data
#Volume of feature input space increases with variables, hence reducing input space has positive effects on reducing time expense of ML algorithms
# Variance preservation is practiced in such a way that highly correlated features are merged
from sklearn.decomposition import PCA



def PCA_accuracy_time_study(train_x, train_y, test_x, test_y, shrink_by):
    original_feature_vector_x=train_x.shape[1]    
    feature_vector= original_feature_vector_x//shrink_by  #reducing input space by shrink factor 

#applying dimensionality reduction on training and testing data
    pca_train_x = PCA(n_components=feature_vector)
    pca_train_x.fit(train_x)
    train_x_pca = pca_train_x.transform(train_x)
    print("original shape:   ", train_x.shape)
    print("transformed shape:", train_x_pca.shape)
    
    pca_test_x = PCA(n_components=feature_vector)
    pca_test_x.fit(test_x)
    test_x_pca = pca_test_x.transform(test_x)
    print("original shape:   ", test_x.shape)
    print("transformed shape:", test_x_pca.shape)
    
    K_max=7
    
    start_time = time.time()
#generating training and testing scoring for KNN classification for dimensionally reduced data with PCA    
    score_train, score_test = KNN_leaveOneOut_score(train_x_pca, train_y, test_x_pca, test_y, K_max=7)
    import matplotlib.pyplot as plt


    plt.plot(range(1,K_max+1), score_train, label='Train')
    plt.plot(range(1,K_max+1), score_test, label='Test')
    plt.legend()
    plt.xlabel('K values')
    plt.ylabel('KNN LeaveOneOutscore score')
    time_taken=time.time()-start_time
    run_time[str(feature_vector)]=time_taken
    plt.title("Time taken for this run: "+str(time_taken)+"PCA n: "+str(feature_vector))
    plt.show()
    










# In[277]:


#Keeping feature size to 784
PCA_accuracy_time_study(train_x, train_y, test_x, test_y, shrink_by=1)


# In[276]:



#Reducing feature size from 784 to 392
PCA_accuracy_time_study(train_x, train_y, test_x, test_y, shrink_by=2)



# In[278]:


#Reducing feature size from 784 to 156
PCA_accuracy_time_study(train_x, train_y, test_x, test_y, shrink_by=5)


# In[280]:


#Reducing feature size from 784 to 78
PCA_accuracy_time_study(train_x, train_y, test_x, test_y, shrink_by=10)


# In[291]:


784//700




# In[282]:


#Reducing feature size from 784 to 39
PCA_accuracy_time_study(train_x, train_y, test_x, test_y, shrink_by=20)


# In[285]:


#Reducing feature size from 784 to 19
PCA_accuracy_time_study(train_x, train_y, test_x, test_y, shrink_by=40)


# In[287]:


#Reducing feature size from 784 to 9
PCA_accuracy_time_study(train_x, train_y, test_x, test_y, shrink_by=80)


# In[289]:


#Reducing feature size from 784 to 4
PCA_accuracy_time_study(train_x, train_y, test_x, test_y, shrink_by=160)


# In[293]:


#Reducing feature size from 784 to 1
PCA_accuracy_time_study(train_x, train_y, test_x, test_y, shrink_by=700)



# In[294]:


n_pca=[784, 392, 156, 78, 39, 19, 9, 4 ,1]
time_per_run=[66.5,33.0,10.41,7.58,5.01,3.50,2.86,2.50,3.27]
plt.plot(n_pca, time_per_run)
plt.xlabel("number of PCA dimensions")
plt.ylabel("time taken for KNN classification")
plt.title("Time Study")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




