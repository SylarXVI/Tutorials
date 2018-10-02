
# coding: utf-8

# # Tutorial 1: Basic data handeling with python

# In[ ]:


'''
Content: 
    Getting started
        Importing the requiered libraries with a short reminder/explanation why we need them
    The data
        loading the data from the source and displaying it in a nice, clean way (pandas)
    Visualisation
        Basic visualisation of the data in order to get a better feeling for it
    Feeding it into the algorithem
        
    What do we see?
        
'''    


# #### Getting started

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#import urllib.request
from sklearn import datasets


# In[99]:


np.random.seed(2304)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


#url = 'https://assets.datacamp.com/production/course_1939/datasets/diabetes.csv'
#urllib.request.urlretrieve(url, 'diabetes.csv')
cancer = datasets.load_breast_cancer()
#print(cancer['DESCR'])
print(f"Features: {cancer['feature_names']}")


# In[36]:


#diabetes = pd.read_csv('diabetes.csv')
#diabetes.head()


# In[39]:


print(f"Labels: {cancer['target_names']}")


# In[40]:


cancer['data'].shape


# In[121]:


cancer_dataframe = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
cancer_dataframe.head(5)


# #### What is this data? 
# 
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

# In[133]:


pd.plotting.scatter_matrix(cancer_dataframe, c=cancer['target'], figsize=(30,30), marker='o', s=5)
plt.show()
'Benign is yellow, Malignant is Violet'
#pd.DataFrame.hist(cancer_dataframe, figsize=(20, 20))


# In[136]:


cancer_dataframe.plot.scatter(x='mean area', y='mean symmetry', c=cancer['target'], colormap='viridis', s=50, title='relatively good separation');
cancer_dataframe.plot.scatter(x='mean fractal dimension', y='worst symmetry', c=cancer['target'], colormap='viridis', s=50, title='relatively bad separation');
cancer_dataframe.plot.scatter(x='mean radius', y='mean area', c=cancer['target'], colormap='viridis', s=50, title='contains no useful information since the data is highly correlated');

'''some combinations of features are already well suited for the characterisation of cancer by themselfes. 
This is showcased by scatterplots with relatively clear separation between the two labels'''

'''We can select the best features and feed them into mashine learning model'''


# ###### We can see

# In[178]:


X = cancer['data'][:, [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 26, 27, 28, 29]]
y = cancer['target']

X.shape
#print(y[:300])



# In[172]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(f'Training data: {X_train.shape}\nTest data: {X_test.shape}')


# In[173]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=2304) # random_state for reproducibility
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[174]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[175]:


importances = rf.feature_importances_
importances


# In[214]:


indices = np.argsort(importances)[::-1]
#XX = cancer_dataframe[cancer_dataframe.columns=['mean area'])]
#XX.head(5)
#indices
#for f in range(X.shape[1]): 
 #   print(f'{X.columns[indices[f]]}: {np.round(importances[indices[f]],2)}')
    
list(cancer_dataframe.columns.values)

