#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# STEPS  FOLLOWED:

# 1. DATA CLEANING
# 2. DATA NULL HANDLING, DE-DUPLICATION HANDLING DUPLICATE VALUES etc
# 3. CORRELATION MATRIX SELECT 20 MOST CORRELATED COLUMNS DATA
# 4. FEATURE SELECTION & PERFORM DIMENSION REDUCTION USING PCA 
# 5. USE THE PCA OUTPUT FOR MODEL BUILDING
# 6. MODEL SELECTION, MODEL BUILDING,VALIDATION & FINDING ACCURACY SCORES                 
# 7.HYPER PARAMETER TUNING & MODEL OPTIMIZATION


# In[15]:


#1 Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb
import scipy as sp
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Importing the sklearn libraries
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


# In[53]:


get_ipython().system('pip install xgboost')


# In[16]:


#2 Importing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[17]:


#3 Setting the display properties
pd.set_option('display.max_rows',10500,'display.max_columns',500)


# In[18]:


#4 Load Data
data = pd.read_csv('Q1_data.csv')
data.head()


# In[19]:


#5 Shape of data
data.shape


# In[ ]:


#6 Displaying entire data
data


# In[6]:


#7 Information about the data with (datatypes)
data.info()


# In[7]:


#8 Handling NULL & Missing Values
data.isnull().sum().sum()


# In[8]:


data.isnull().sum()


# In[21]:


#9 Replacing NULL with Average Value 
data.fillna(data.mean(),inplace=True)


# In[22]:


duplicates_ = data.duplicated()
data[duplicates_]


# In[23]:


#Data Correlation - Identifying Correlation between Attributes 
data.corr()


# In[75]:


#Correlation of data with target
corr_output = data.corr()
corr_output.info()


# In[82]:


#Obtaining Correlation output
corr_output


# In[85]:


#Identifying the values that are having highest correlation with the target variable 
top_corr_values = corr_output.nlargest(11, ['target'])
top_corr_values


# In[91]:


Xfs = top_corr_values.loc[:,['target']]
Xfs


# In[18]:


#11 Feature Selection Using Extra Trees
from sklearn.ensemble import ExtraTreesClassifier
plt.figure(figsize=(20,8))
sb.set(font_scale=0.5)
x = data.iloc[:,:-1]
y = data.iloc[:,151:152]
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
print(feat_importances.nlargest(20))
y1 = feat_importances.nlargest(20)
y1.to_csv("Features_Selected.csv")


# In[137]:


feat_selection = feat_importances.nlargest(20)
feat_selection


# In[24]:


#12 Perform Dimension Reduction Using PCA
x = data.iloc[:,:-1]
y = data.target
pca = PCA()
pca_data = pd.DataFrame(pca.fit_transform(data))
pca_data.head()


# In[34]:


x1 = pca_data.iloc[:,:151]
x1.head()


# In[27]:


pca.components_


# In[28]:


pca.n_components_


# In[32]:


#13 Plot PCA output
plt.plot(range(0,152),pca.explained_variance_ratio_)


# In[ ]:


#Model Selection & Model Building


# In[29]:


# 14.1.1 KNN 
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=100)
#kneighbors classifier model
model_knn = KNeighborsClassifier()
#fitting the knn model
model_knn.fit(x_train,y_train)
#predicting the target
y_predict_knn = model_knn.predict(x_test)
print("Classification Model Report for KNN:")
#printing classification report
print(classification_report(y_test,y_predict_knn))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_knn))


# In[30]:


# 14.1.2 KNN using pca
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#kneighbors classifier model
model_knn_pca = KNeighborsClassifier()
#fitting the knn model
model_knn_pca.fit(x_train,y_train)
#predicting the target
y_predict_knn_pca = model_knn_pca.predict(x_test)
#printing classification report
print("Classification Model Report for KNN USING PCA:")
print(classification_report(y_test,y_predict_knn_pca))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_knn_pca))


# In[32]:


# 14.2 Support Vector Classification 
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=100)
#svc classifier model
model_svc = SVC()
#fitting the knn model
model_svc.fit(x_train,y_train)
#predicting the target
y_predict_svc = model_svc.predict(x_test)
#printing classification report
print("Classification Model Report for SVM:")
print(classification_report(y_test,y_predict_svc))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_svc))


# In[35]:


# 14.2 Support Vector Classification using pca
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#svc classifier model
model_svc_pca = SVC()
#fitting the svc model
model_svc_pca.fit(x_train,y_train)
#predicting the target
y_predict_svc_pca = model_svc.predict(x_test)
#printing classification report
print("Classification Model Report for SVM USING PCA:")
print(classification_report(y_test,y_predict_svc_pca))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_svc_pca))


# In[36]:


# 14.3 Decision Tree Classifier
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#dt classifier model
model_dt = DecisionTreeClassifier()
#fitting the dt model
model_dt.fit(x_train,y_train)
y_predict_dt = model_dt.predict(x_test)
#printing classification report
print("Classification Model Report for DECISION TREE:")
print(classification_report(y_test,y_predict_dt))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_dt))


# In[37]:


# 14.3 Decision Tree Classifier using pca
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#dt classifier model
model_dt_pca = DecisionTreeClassifier()
#fitting the dt model
model_dt_pca.fit(x_train,y_train)
#predicting the target
y_predict_dt_pca = model_dt_pca.predict(x_test)
#printing classification report
print("Classification Model Report for DECISION TREE USING PCA:")
print(classification_report(y_test,y_predict_dt_pca))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_dt_pca))


# In[38]:


# 14.4 Random Forest Classifier
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#rf classifier model
model_rf = RandomForestClassifier()
#fitting the rf model
model_rf.fit(x_train,y_train)
#predicting the target
y_predict_rf = model_rf.predict(x_test)
#printing classification report
print("Classification Model Report for RANDOM FOREST:")
print(classification_report(y_test,y_predict_rf))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_rf))


# In[39]:


# 14.4 Random Forest Classifier using pca
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#rf classifier model
model_rf_pca = RandomForestClassifier()
#fitting the rf model
model_rf_pca.fit(x_train,y_train)
#predicting the target
y_predict_rf_pca = model_rf_pca.predict(x_test)
#printing classification report
print("Classification Model Report for RANDOM FOREST USING PCA:")
print(classification_report(y_test,y_predict_rf_pca))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_rf_pca))


# In[40]:


# 14.5 Gradient Boost Classifier
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#gb classifier model
model_gb = GradientBoostingClassifier()
#fitting the gb model
model_gb.fit(x_train,y_train)
#predicting the target
y_predict_gb = model_gb.predict(x_test)
#printing classification report
print("Classification Model Report for GRADIENT BOOST:")
print(classification_report(y_test,y_predict_gb))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_gb))


# In[41]:


# 14.5 Gradient Boost Classifier using pca
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#gb classifier model
model_gb_pca = GradientBoostingClassifier()
#fitting the gb model
model_gb_pca.fit(x_train,y_train)
#predicting the target
y_predict_gb_pca = model_gb_pca.predict(x_test)
#printing classification report
print("Classification Model Report for GRADIENT BOOST USING PCA:")
print(classification_report(y_test,y_predict_gb_pca))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_gb_pca))


# In[234]:


# 14.7 XG Boost Classifier
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#xgboost classifier model
model_xgb = XGBClassifier()
#fitting the xgboost model
model_xgb.fit(x_train,y_train)
#predicting the target
y_predict_xgb = model_xgb.predict(x_test)
#printing classification report
print("Classification Model Report for XG BOOST:")
print(classification_report(y_test,y_predict_xgb))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_xgb))


# In[49]:


# 14.7 XG Boost Classifier using pca
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#xgboost classifier model
model_xgb_pca = XGBClassifier()
#fitting the xgboost model
model_xgb_pca.fit(x_train,y_train)
#predicting the target
y_predict_xgb_pca = model_xgb_pca.predict(x_test)
#printing classification report
print("Classification Model Report for XG BOOST USING PCA:")
print(classification_report(y_test,y_predict_xgb_pca))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_xgb_pca))


# In[236]:


# 14.8 ANN Classifier
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#ann classifier model
model_ann = MLPClassifier()
#fitting the xgboost model
model_ann.fit(x_train,y_train)
#predicting the target
y_predict_ann = model_ann.predict(x_test)
#printing classification report
print("Classification Model Report for ANN:")
print(classification_report(y_test,y_predict_ann))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_ann))


# In[44]:


# 14.8 ANN Classifier using pca
#splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y,random_state=100)
#ann classifier model
model_ann_pca = MLPClassifier()
#fitting the ann model
model_ann_pca.fit(x_train,y_train)
#predicting the target
y_predict_ann_pca = model_ann_pca.predict(x_test)
#printing classification report
print("Classification Model Report for ANN:")
print(classification_report(y_test,y_predict_ann_pca))
#calculating accuracy score
print("Accuracy Score",accuracy_score(y_test,y_predict_ann_pca))


# In[238]:


#Evaluating Accuracy of models and comparing them to select the best model
performance_df = pd.DataFrame(columns=['Models','Accuracy'])
performance_df['Models'] = ['KNN','SVM','DecisionTree','RandomForest','GradientBoost','XGBoost','ANN']
performance_df['Accuracy'] = [accuracy_score(y_test,y_predict_knn),accuracy_score(y_test,y_predict_svc),accuracy_score(y_test,y_predict_dt),accuracy_score(y_test,y_predict_rf),accuracy_score(y_test,y_predict_gb),accuracy_score(y_test,y_predict_xgb),accuracy_score(y_test,y_predict_ann)]
performance_df = performance_df[['Models','Accuracy']]


# In[239]:


#Evaluating Accuracy of models and comparing them to select the best model
print("The Accuracy Comparison Chart before pca is as below:\n\n",performance_df)


# In[240]:


#Evaluating Accuracy of models and comparing them to select the best model using graph 
import seaborn as sns
sns.set(rc={'figure.figsize':(5,5)})
sns.set(font_scale = 1.5)
sns.barplot(x='Models',y='Accuracy',data=performance_df.sort_values(by='Accuracy'), palette="Greens_r")
plt.xticks(rotation='vertical')
plt.show()


# In[241]:


# Evaluating Accuracy of models and comparing them to select the best model  after using pca
performance_df_pca = pd.DataFrame(columns=['Models','Accuracy'])
performance_df_pca['Models'] = ['KNN','SVM','DecisionTree','RandomForest','GradientBoost','XGBoost','ANN']
performance_df_pca['Accuracy'] = [accuracy_score(y_test,y_predict_knn_pca),accuracy_score(y_test,y_predict_svc_pca),accuracy_score(y_test,y_predict_dt_pca),accuracy_score(y_test,y_predict_rf_pca),accuracy_score(y_test,y_predict_gb_pca),accuracy_score(y_test,y_predict_xgb_pca),accuracy_score(y_test,y_predict_ann_pca)]
performance_df_pca = performance_df_pca[['Models','Accuracy']]


# In[244]:


# Evaluating Accuracy of models and comparing them to select the best model  after using pca
print("The Accuracy Comparison Chart after pca is as below:\n\n",performance_df_pca)


# In[245]:


# Evaluating Accuracy of models and comparing them to select the best model  after using pca using graph
import seaborn as sns
sns.set(rc={'figure.figsize':(5,5)})
sns.set(font_scale = 1.5)
sns.barplot(x='Models',y='Accuracy',data=performance_df_pca.sort_values(by='Accuracy'), palette="Greens_r")
plt.xticks(rotation='vertical')
plt.show()


# In[246]:


#using cross-validation technique instead of train and split


# In[303]:


#KNN 
scores_knn = cross_val_score(model_knn_pca,x1,y,cv=10,scoring='accuracy')
print(scores_knn)
print("Average Accuracy:",scores_knn.mean())
print("std value:", scores_knn.std())
#std < 0.05 is a good model


# In[304]:


#SVM
scores_svm = cross_val_score(model_svc_pca,x1,y,cv=10,scoring='accuracy')
print(scores_svm)
print("Average Accuracy:",scores_svm.mean())
print("std value:", scores_svm.std())
#std < 0.05 is a good model


# In[305]:


#Decision tree
scores_dt = cross_val_score(model_dt_pca,x1,y,cv=10,scoring='accuracy')
print(scores_dt)
print("Average Accuracy:",scores_dt.mean())
print("std value:", scores_dt.std())
#std < 0.05 is a good model


# In[306]:


#Random Forest
scores_rf = cross_val_score(model_rf_pca,x1,y,cv=10,scoring='accuracy')
print(scores_rf)
print("Average Accuracy:",scores_rf.mean())
print("std value:", scores_rf.std())
#std < 0.05 is a good model


# In[ ]:


#Gradient Boost
scores_gb = cross_val_score(model_gb_pca,x1,y,cv=10,scoring='accuracy')
print(scores_gb)
print("Average Accuracy:",scores_gb.mean())
print("std value:", scores_gb.std())
#std < 0.05 is a good model


# In[312]:


scores_xgb = cross_val_score(model_xgb_pca,x1,y,cv=10,scoring='accuracy')
print(scores_xgb)
print("Average Accuracy: ")
print("std value: 0.002956956990097")
#std < 0.05 is a good model0.9775411585365854


# In[307]:


#XG Boost
scores_xgb = cross_val_score(model_xgb_pca,x1,y,cv=10,scoring='accuracy')
print(scores_xgb)
print("Average Accuracy:",scores_xgb.mean())
print("std value:", scores_xgb.std())
#std < 0.05 is a good model


# In[308]:


#ANN
scores_ann = cross_val_score(model_ann_pca,x1,y,cv=10,scoring='accuracy')
print(scores_ann)
print("Average Accuracy:",scores_ann.mean())
print("std value:", scores_ann.std())
#std < 0.05 is a good model


# In[326]:


# Evaluating Accuracy of models and comparing them to select the best model  cross  -validation & pca
performance_df_cvk = pd.DataFrame(columns=['Models','Accuracy'])
performance_df_cvk['Models'] = ['KNN','SVM','DecisionTree','RandomForest','GradientBoost','XGBoost','ANN']
performance_df_cvk['Accuracy'] = [scores_knn.mean(),scores_svm.mean(),scores_dt.mean(),
                                 scores_rf.mean(),
                                 scores_gb.mean(),
                                 scores_xgb.mean(),
                                 scores_ann.mean()]
performance_df_cv['Std Scores'] = [scores_knn.std(),scores_svm.std(),scores_dt.std(),
                                 scores_rf.std(),
                                 scores_gb.std(),
                                 scores_xgb.std(),
                                 scores_ann.std()]                                
performance_df_cv = performance_df_cv[['Models','Accuracy']]


# In[327]:


# Evaluating Accuracy of models and comparing them to select the best model  cross  -validation & pca
print("The Accuracy Comparison Chart using k fold cross validation after pca is as below:\n\n",performance_df_cvk)


# In[ ]:


#PERFORMANCE OPTIMIZATION OR HYPER PARAMETER TUNING


# In[290]:


from sklearn.model_selection import RandomizedSearchCV
#data_hp = pca_data.reshape(-1, 1)
x1 = data.iloc[:,:10]
parameters = {'max_depth':[15,16,18,19,20],
              'random_state': [1,2,3,4,5],
              'n_estimators':[80,90,100,110,120]}
rand = RandomizedSearchCV(model_rf_pca,parameters,cv=5)
rand.fit(x1,y)
print(rand.best_score_)
print(rand.best_params_)


# In[291]:


from sklearn.model_selection import RandomizedSearchCV
#data_hp = pca_data.reshape(-1, 1)
x1 = pca_data.iloc[:,:10]
parameters = {'random_state': [1,2,3,4,5]}
rand = RandomizedSearchCV(model_svc_pca,parameters,cv=5)
rand.fit(x1,y)
print(rand.best_score_)
print(rand.best_params_)


# In[42]:


from sklearn.model_selection import RandomizedSearchCV
#data_hp = pca_data.reshape(-1, 1)
x1 = pca_data.iloc[:,:10]
parameters = {'random_state': [1,2,3,4,5]}
rand = RandomizedSearchCV(model_svc_pca,parameters,cv=5)
rand.fit(x1,y)
print(rand.best_score_)
print(rand.best_params_)


# In[45]:


from sklearn.model_selection import RandomizedSearchCV
#data_hp = pca_data.reshape(-1, 1)
x1 = pca_data.iloc[:,:10]
parameters = {'random_state': [1,2,3,4,5]}
rand = RandomizedSearchCV(model_ann_pca,parameters,cv=5)
rand.fit(x1,y)
print(rand.best_score_)
print(rand.best_params_)


# In[46]:


from sklearn.model_selection import RandomizedSearchCV
x1 = pca_data.iloc[:,:10]
parameters = {'random_state': [1,2,3,4,5]}
rand = RandomizedSearchCV(model_dt_pca,parameters,cv=5)
rand.fit(x1,y)
print(rand.best_score_)
print(rand.best_params_)


# In[47]:


from sklearn.model_selection import RandomizedSearchCV
#data_hp = pca_data.reshape(-1, 1)
x1 = pca_data.iloc[:,:10]
parameters = {'random_state': [1,2,3,4,5]}
rand = RandomizedSearchCV(model_gb_pca,parameters,cv=5)
rand.fit(x1,y)
print(rand.best_score_)
print(rand.best_params_)


# In[50]:


from sklearn.model_selection import RandomizedSearchCV
#data_hp = pca_data.reshape(-1, 1)
x1 = pca_data.iloc[:,:10]
parameters = {'random_state': [1,2,3,4,5]}
rand = RandomizedSearchCV(model_xgb_pca,parameters,cv=5)
rand.fit(x1,y)
print(rand.best_score_)
print(rand.best_params_)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
x1 = pca_data.iloc[:,:10]
parameters = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
rand = RandomizedSearchCV(model_svc_pca,parameters,cv=5)
rand.fit(x1,y)
print("\n the best score is\n\n", rand.best_score_)
print(rand.best_params_)


# In[299]:


DecisionTreeClassifier(random_state=4).get_params().keys()


# In[336]:


SVC(random_state=1).get_params().keys()


# In[ ]:



param_grid = {'C': [0.1,1, 10, 100], 
'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
print(grid.best_estimator_)

