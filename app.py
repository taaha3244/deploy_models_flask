#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install flask')


# In[10]:


import numpy as np
from flask import Flask,request,render_template
import pickle


# In[11]:


import os

# Replace 'path/to/your/directory' with the path of the directory you want to set as the new CWD
os.chdir("D:\deploy_models_flask")


# In[12]:


app=Flask(__name__)


# In[13]:


model=pickle.load(open('models/model.pkl','rb'))


# In[14]:


model


# In[15]:


@app.route('/')
def home():
    return render_template('index.html')


# In[16]:


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    features=[np.array(int_features)]
    prediction=model.predict(features)
    output=round(prediction[0],2)
    return render_template('index.html',prediction_text='Percent with heart disease is:{}'.format(output))
    


# In[ ]:


if __name__=="__main__":
    app.run()


# In[ ]:




