#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc,accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,classification_report


# In[2]:


ds = pd.read_csv("drybeans.csv")


# In[3]:


ds.isnull().count()


# In[4]:


ds.head()


# In[5]:


#checking all the quantities for the given data
ds.describe()


# In[6]:


ds.shape


# In[7]:


ds.duplicated().sum() #checking for duplicates


# In[8]:


df = ds.drop_duplicates() #removing duplicates and saving the data as another dataset


# In[9]:


df.shape 


# In[10]:


df.corr()


# In[11]:


# train and test split of data
x = df.drop(['Class'],axis = 1)
y = df['Class']

x_train_data,x_test_data, y_train_data,y_test_data = train_test_split(x, y , train_size=.80, random_state=50, stratify=df.Class)


# In[12]:


#identifying and storing the correlated features so as to drop later
correlated_features = ['Area', 'EquivDiameter', 'MajorAxisLength', 'MinorAxisLength','ShapeFactor3','AspectRation','Eccentricity']
#Area - highly correlated with ConvexArea and ConvexArea has more Positive correlation with other features than Area.(99.9% correlation)
#EquivDiameter - # highly correlated with ConvexArea.(98.4% correlation) - also correlated with Perimeter.
#MajorAxisLength - it is highly correlated with perimeter - #Perimeter is not dropped as it is easily calculable when compared.
#MinorAxisLength -  correlated with ConvexArea(95%)
#ShapeFactor3 - corrleated with Compactness
#AspectRatio - correlated with compactness
#Eccentricity - correlated with compactness


# In[13]:


#using standard scaler to scale the features

scaler = StandardScaler()
x_train_data = pd.DataFrame(scaler.fit_transform(x_train_data), columns = x_train_data.columns)
x_test_data = pd.DataFrame(scaler.transform(x_test_data), columns = x_test_data.columns)

# Changing target variable from string to integers randing from n to n-1
encoding_target = LabelEncoder()
y_train_data = encoding_target.fit_transform(y_train_data)
y_test_data = encoding_target.transform(y_test_data)


# In[14]:


#it seems dropping the correlated features decreases the accuracy of the given training set slightly.
#Dropping the correlated features for the test and train set.
x_train_final = x_train_data.drop(correlated_features,axis=1)
x_test_final = x_test_data.drop(correlated_features,axis=1)


# In[15]:


#PART 2

# Training:
#1-Penalty - None, class_weight = None
log_no_weight = LogisticRegression(penalty = 'none', class_weight=None)
log_no_weight.fit(x_train_final,y_train_data)


# In[16]:


# Accuracy of the model
accuracy_score(y_test_data,log_no_weight.predict(x_test_final))


# In[17]:


balanced_accuracy_score(y_test_data,log_no_weight.predict(x_test_final))


# In[18]:


#2-Penalty - None, class_weight = balanced

log_balanced = LogisticRegression(penalty = 'none', class_weight='balanced')
log_balanced.fit(x_train_final,y_train_data)


# In[19]:


# Accuracy of the model
accuracy_score(y_test_data,log_balanced.predict(x_test_final))


# In[20]:


#using predict, decision function, predict_probab
log_no_weight.predict(x_test_final)


# In[21]:


log_no_weight.decision_function(x_test_final)


# In[22]:


log_no_weight.predict_proba(x_test_final)


# In[23]:


#Evaluation

# Accuracy from the above model
accuracy_score(y_test_data,log_balanced.predict(x_test_final))


# In[24]:


# Balanced accuracy
balanced_accuracy_score(y_test_data,log_balanced.predict(x_test_final))


# In[25]:


# Confusion matrix for unbalanced
conf_matrix_unbalanced = confusion_matrix(y_test_data,log_no_weight.predict(x_test_final))
conf_matrix_unbalanced


# In[26]:


#confusion matrix for the balanced
conf_matrix_balanced = confusion_matrix(y_test_data,log_balanced.predict(x_test_final))
conf_matrix_balanced


# In[27]:


# Using classification report we can check required metrics for last model
print(classification_report(y_test_data,log_balanced.predict(x_test_final)))


# In[ ]:





# In[28]:


#Advanced section

# Training - penalty with class weights as balanced
log_balanced_with_penalty = LogisticRegression(penalty = 'l2', class_weight='balanced')
log_balanced_with_penalty.fit(x_train_final,y_train_data)


# In[29]:


accuracy_score(y_test_data,log_balanced_with_penalty.predict(x_test_final))


# In[30]:


# Confusion matrix for unbalanced
conf_matrix_balanced_penalty = confusion_matrix(y_test_data,log_balanced_with_penalty.predict(x_test_final))
conf_matrix_balanced_penalty

