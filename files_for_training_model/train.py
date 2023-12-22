#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


import os

# Replace 'path/to/your/directory' with the path of the directory you want to set as the new CWD
os.chdir("D:\deploy_models_flask")


# In[6]:


df=pd.read_csv("heart_data.csv")


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df = df.drop(columns=["Unnamed: 0"])


# In[20]:


sns.lmplot(x='biking', y='heart.disease', data=df)  
sns.lmplot(x='smoking', y='heart.disease', data=df)  
plt.show()


# In[21]:


x_df = df.drop('heart.disease', axis=1)
y_df = df['heart.disease']


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)


# In[23]:


from sklearn import linear_model

#Create Linear Regression object
model = linear_model.LinearRegression()

#Now let us call fit method to train the model using independent variables.
#And the value that needs to be predicted (Images_Analyzed)

model.fit(X_train, y_train) #Indep variables, dep. variable to be predicted
print(model.score(X_train, y_train))  #Prints the R^2 value, a measure of how well


# In[24]:


from sklearn.metrics import mean_squared_error, r2_score

prediction_test = model.predict(X_test)    
mse = mean_squared_error(y_test, prediction_test)
r2 = r2_score(y_test, prediction_test)
mse,r2


# In[25]:


import pickle
pickle.dump(model, open('model.pkl','wb'))



# In[26]:


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4.5, 100.3]]))


#Model is ready. Let us check the coefficients, stored as reg.coef_.
#These are a, b, and c from our equation. 
#Intercept is stored as reg.intercept_
#print(model.coef_, model.intercept_)

#All set to predict the number of images someone would analyze at a given time
#print(model.predict([[13, 2, 23]]))


# In[ ]:




