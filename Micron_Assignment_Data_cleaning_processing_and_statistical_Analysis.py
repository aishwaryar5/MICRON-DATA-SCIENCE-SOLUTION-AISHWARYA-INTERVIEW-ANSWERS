#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DATA CLEANING AND PROCESSING DATA 


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import statistics
import time
#STATISTICAL ANALYSIS USING PYTHON
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.gridspec import GridSpec
pd.set_option('display.max_columns', 100)
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
import json
import requests


# In[3]:


data = pd.read_csv('Q1_data.csv')


# In[54]:


from pandas import plotting


# In[4]:


pd.set_option('display.max_rows',10500,'display.max_columns',500)


# In[ ]:


#UNDERSTANDING DATA 


# In[5]:


data.info()


# In[72]:


data.shape


# In[73]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.isnull().sum().sum()


# In[ ]:


data.isna().sum()
data.isna().sum().sum()


# In[9]:


data.shape


# In[37]:


data.describe()


# In[39]:


CORR_DATA2 = data.loc[:,['64','56','9','0','133','target']]
CORR_DATA2


# In[40]:


CORR_DATA2.describe()


# In[13]:


#Replacing NULL with Average Value 
data.fillna(data.mean(),inplace=True)


# In[10]:


x=data.duplicated()
data[x]


# In[16]:


#Correlation matrix 
corr_output= data.corr()


# In[12]:


data.corr()


# In[17]:


#Identifying the values that are having highest correlation with the target variable 
top_corr_values = corr_output.nlargest(11, ['target'])
Xfs = top_corr_values.loc[:,['target']]
Xfs


# In[20]:


#WE FIND THAT THE TOP HIGHEST CORRELATED COLUMNS ARE :
CORR_DATA = data.loc[:,['64','56','9','0','133','25','108','50','100','11','target']]
CORR_DATA


# In[26]:


#the top 5 highest correlated columns are
CORR_DATA2 = data.loc[:,['64','56','9','0','133','target']]
CORR_DATA2


# In[28]:


correlation_matrix = CORR_DATA2.corr()


# In[29]:


plt.figure(figsize=(18,8))
sns.set(font_scale=0.5)
cmap = sns.light_palette("#800080",as_cmap=True)
sns.heatmap(correlation_matrix, cmap=cmap,annot=True)
plt.title("Correlation Matrix",fontsize=19)
plt.savefig('plot16.png', dpi=300, bbox_inches='tight')
plt.show()


# In[48]:


CORR_DATA2.to_excel('Highest_correlated_data.xlsx')


# In[50]:


data_after_filter = pd.read_excel('Highest_correlated_data.xlsx')
data_after_filter


# In[14]:


data.nunique()


# In[24]:


data['target'].unique()


# In[25]:


data['target'].nunique()


# In[34]:


data['target'].value_counts()


# In[42]:


data['target'].cummin()


# In[32]:


data['64'].nunique()


# In[33]:


data['64'].unique()


# In[35]:


data['64'].value_counts()


# In[36]:


data['56'].value_counts()


# In[43]:


Corr.skew()


# In[63]:


skewness = CORR_DATA2.skew()


# In[74]:


skewness2 = data.skew()


# In[75]:


skewness2.plot()


# In[65]:


CORR_DATA2.skew()


# In[64]:


skewness.plot()


# In[45]:


CORR_DATA2.kurtosis()


# In[66]:


kurtosis = CORR_DATA2.kurtosis()


# In[76]:


kurtosis2 = data.kurtosis()


# In[67]:


kurtosis.plot()


# In[77]:


kurtosis2.plot()


# In[46]:


CORR_DATA2.sample()


# In[58]:


from pandas.plotting import scatter_matrix
plotting.scatter_matrix(CORR_DATA2[['64', '56', '9','0','133','target']])


# In[60]:


plotting.scatter_matrix(CORR_DATA2[['64', '56','target']])


# In[61]:


plotting.scatter_matrix(CORR_DATA2[['9', '0','target']])


# In[68]:


plotting.scatter_matrix(CORR_DATA2[['9', '0','target']])  


# In[ ]:




