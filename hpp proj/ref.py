#!/usr/bin/env python
# coding: utf-8

# HOUSE PRICE PREDICTION 

# In[28]:


# importing requires libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics # this is used atlast for calc the accuracy
import numpy as np


# LOADING DATASET

# In[12]:


# dataset we chose is us house details
data = pd.read_csv("D:\\datascience\\hpp proj\\USA_Housing.csv") # we store our dataset in the var called data
data.head() #this basically has 7 columns
#linear reg is a mathematical model which predicts the value of the model so all the col are in float and int except the address so we drop that column


# DROP THE ADDRESS COLUMN

# In[17]:


data = data.drop(['Address'],axis=1)
data.head()


# CHECK FOR MISSING DATA

# In[19]:


#0inorder to find this we r gng to plot the heatmap
sns.heatmap(data.isnull())  #so if there is any missing entity it will be showed in the heatmap


# TRAIN TEST SPLIT

# In[22]:


# here we r gng to split the dataset into train and test split

#first step is seperating ip and op variables seperately
X = data.drop(['Price'],axis=1)  #this contains all the input variables
Y = data['Price']  #this is the oput var

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.30)  #here we use train data for training and test datv for testing

#0.3 for test and 0.7 for train


# TRAINING AND PREDICTING 

# In[23]:


# lets create a var called model and fit the train data in it

model = LinearRegression()
model.fit(X_train,Y_train)  #traing the model


# In[26]:


predictions = model.predict(X_test) #testing the model
predictions


# EVALUATION OF MODEL

# In[32]:


# now we r gng to evaluate our model using rms error
error = np.sqrt(metrics.mean_absolute_error(Y_test, predictions))  #y_test has actual result and prediction has our model trained results
error


# In[33]:


# so in this way we have trained our model and test , but in real case user will give inputs and our model should make predictions accordingly


# In[ ]:




