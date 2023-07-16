#!/usr/bin/env python
# coding: utf-8

# In[4]:


# PART 1: DATA PREPROCESSING
#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


# READING DATASET
data=pd.read_csv('data.csv')


# In[6]:


data.head()


# In[7]:


## PART 1.2: DATA EXPLORATION


# In[8]:


data.shape


# In[9]:


data.info()


# In[10]:


# Getting the no. of categorical data(variable) 
data.select_dtypes(include='object').columns


# In[11]:


#Length of categorical variable
len(data.select_dtypes(include='object').columns)


# In[12]:


# Printing all numerical variable
data.select_dtypes(include=['float64','int64']).columns


# In[13]:


# statistical summary
data.describe()


# In[14]:


#List of columns
data.columns


# In[15]:


# PART 1.3: DEALING WITH THE MISSING VALUES


# In[16]:


# Checking whether there are null value or not
data.isnull().values.any()
# True means null values present


# In[17]:


# Calculating the number of null values
data.isnull().values.sum()


# In[18]:


# Getting the columns containing null values
data.columns[data.isnull().any()]


# In[19]:


len(data.columns[data.isnull().any()])
# Output 1 means only one columns with null values


# In[20]:


# Checking the number of non null values in the Unnamed column
data['Unnamed: 32'].count()
# 0 means there are no any non null values
# So we can drop the entire columns


# In[21]:


data = data.drop(columns='Unnamed: 32')


# In[22]:


data.shape
# initially we had 33 columns now we have only 32


# In[23]:


# Again checking for the null values(variables)
data.isnull().values.any()
# False means no any null values


# In[24]:


# PART 1.4: DEALING WITH CATEGORICAL DATA


# In[25]:


# Printing the categorial data(columns)
data.select_dtypes(include='object').columns


# In[26]:


# Only one categorical columns


# In[27]:


# Printing the unique number of data present in the diagnosis columns
data['diagnosis'].unique()


# In[28]:


# The diagnosis of breast tissues (M = malignant, B = benign)


# In[29]:


# Printing the number of unique data using nunique() fn
data['diagnosis'].nunique()


# In[30]:


# One Hot Encoding is a common way of preprocessing categorical features for machine learning models. 
data = pd.get_dummies(data=data, drop_first=True)


# In[31]:


data.head()


# In[32]:


# PART 1.5: COUNTPLOT


# In[33]:


sns.countplot(data['diagnosis_M'], label='Count')
plt.show()


# In[34]:


# Printing the exact count
# B (0) values
(data.diagnosis_M == 0).sum()


# In[35]:


# There are total 357 benign values


# In[36]:


# Printing the exact count
# M (1) values
(data.diagnosis_M == 1).sum()


# In[37]:


# 212 malignant values


# In[38]:


# PART 1.6: CORRELATION MATRIX AND HEATMAP


# In[39]:


# Removing the dependent variable diagnosis_M
dataset = data.drop(columns='diagnosis_M')


# In[40]:


dataset.head()


# In[41]:


#Finding the correlation between the dependent variables and the independent variables
dataset.corrwith(data['diagnosis_M']).plot.bar(
    figsize=(20,10), title='Correlated with diagnosis_M',rot=45, grid= True
)


# In[42]:


# Plot in +ve axis means independent and dependent variable are related positively
# If -ve then related -vely
# If small bar or no bar then they are not related 


# In[43]:


# CORRELATION MATRIX
corr = data.corr()


# In[44]:


corr


# In[45]:


# Analyzing this data is difficult.
# To solve this problem we will be using HEATMAP
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True)


# In[46]:


# Light colour indicates more correlation 


# In[47]:


# PART 1.7: SPLITTING THE DATASET TRAIN AND TEST SET


# In[48]:


data.head()


# In[49]:


# matrix of feature / independent variables
# : specifies all the rows
# need to include from radius_maen till last excluding the last columns i.e., "diagnosis_M"
x = data.iloc[:,1:-1].values


# In[50]:


x.shape


# In[51]:


# Target / Dependent variable
# Selecting all the rows and only one columns i.e., diagnosis_M
y = data.iloc[:,-1].values
# Here y is a vector


# In[52]:


y.shape


# In[53]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[54]:


x_train.shape


# In[55]:


x_test.shape


# In[56]:


y_train.shape


# In[57]:


y_test.shape


# In[58]:


# PART 1.8: FEATURE SCALING
# We are applying feature scaling because we want all the variables in a same scale


# In[59]:


from sklearn.preprocessing import StandardScaler
# Standardize features by removing the mean and scaling to unit variance.
# The standard score of a sample x is calculated as:

# z = (x - u) / s
# s-> standard deviation of the training sample


# In[60]:


sc = StandardScaler()


# In[61]:


x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[62]:


x_train


# In[63]:


x_test


# In[64]:


# PART 2: BUILDING THE MODEL 


# In[65]:


# PART 2.1: LOGISTIC REGRESSION


# In[66]:


# Goto ofiicial website of sci-kit, goto api, goto linear model and select the logistic regression(penalty)
# For more details refer the official website
from sklearn.linear_model import LogisticRegression
# random_state simply gives the same output
classifier_lr = LogisticRegression(random_state=0) 


# In[67]:


# Now training the logistic regression data using fit() method
classifier_lr.fit(x_train,y_train)


# In[68]:


# prediction
y_pred = classifier_lr.predict(x_test)


# In[69]:


# PREFORMANCES
# From official website goto classification metrics and importing various classes.
# CONFUSION MATRIX is used to check the number of correct score and number of uncorrect score
# ACCURACY SCORE will calculate the accuracy of our model.


# In[70]:


# The F1 SCORE can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. 
# The relative contribution of precision and recall to the F1 score are equal. 
# The formula for the F1 score is:
# F1 = 2 * (precision * recall) / (precision + recall)


# In[71]:


# The RECALL RATIO is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
# The RECALL RATIO is intuitively the ability of the classifier to find all the positive samples.
# The best value is 1 and the worst value is 0.


# In[72]:


# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. 
# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
# The best value is 1 and the worst value is 0.


# In[73]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# In[74]:


acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)


# In[75]:


results = pd.DataFrame([['Logistic Regression', acc, f1, prec, rec]],
                      columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])


# In[76]:


results


# In[77]:


# Plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[78]:


# Here (65,45) are the correct prediction and (2,2) are the wrong prediction


# In[79]:


# PART 2.1.1: CROSS VALIDATION


# In[80]:


# For cross validation goto official website and goto MODEL VALIDATION
from sklearn.model_selection import cross_val_score


# In[81]:


# This cross validation technique will compute 10 different accuracies on the basis of x_train and y_train.
# And every time it will take random input from x_train and y_train
accuracies = cross_val_score(estimator=classifier_lr, X=x_train, y=y_train, cv=10)


# In[82]:


print("Accuracies is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[83]:


# PART 2.2: RANDOM FOREST


# In[84]:


from sklearn.ensemble import RandomForestClassifier


# In[85]:


# creating instance of the class
classifier_rm = RandomForestClassifier(random_state=0)


# 

# In[ ]:





# In[86]:


# Now training the Random Forest data using fit() method
classifier_rm.fit(x_train, y_train)


# In[87]:


# prediction
y_pred = classifier_rm.predict(x_test)


# In[88]:


# Similar to Logistic Regression we need to import all the classes
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# In[89]:


acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)


# In[90]:


model_results = pd.DataFrame([['Random Forest', acc, f1, prec, rec]],
                      columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])


# In[91]:


results = results.append(model_results, ignore_index=True)


# In[92]:


results


# In[93]:


# Plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[94]:


# CROSS VALIDATION
# To decide the final result we are using CV
from sklearn.model_selection import cross_val_score


# In[95]:


accuracies = cross_val_score(estimator=classifier_rm, X=x_train, y=y_train, cv=10)


# In[96]:


print("Accuracies is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[97]:


# SO THE LOGISTIC REGRESSION HAS THE MORE ACCURACY AND LESS SD
# THEREFORE WE ARE CHOOSING THE LOGISTIC REGRESSION MODEL


# In[ ]:





# In[ ]:





# In[ ]:





# In[98]:


# PART 3: RANDOMIZED SEARCH TO FIND THE BEST PARAMETER (LOGISTIC REGRESSION)


# In[99]:


from sklearn.model_selection import RandomizedSearchCV


# In[100]:


# Now we need to define the parameter
# For goto "sklearn.linear_model.LogisticRegression" this module


# In[101]:


parameters = {'penalty':['l1', 'l2', 'elasticnet', 'none'],
              'C':[0.25,0.5,0.75,1.0,1.25,1.50,1.75,2.0], #CONSTANT
              'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    
}


# In[102]:


parameters


# In[103]:


# Defining the instance of the class
random_search = RandomizedSearchCV(estimator=classifier_lr, param_distributions=parameters, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)


# In[104]:


# Tune the parameter
random_search.fit(x_train,y_train)


# In[105]:


random_search.best_estimator_


# In[106]:


random_search.best_score_


# In[107]:


random_search.best_params_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[108]:


# PART 4: FINAL MODEL(LOGISTIC REGRESSION)


# In[109]:


from sklearn.linear_model import LogisticRegression
# random_state simply gives the same output
classifier= LogisticRegression(random_state=0) 
classifier.fit(x_train,y_train)


# In[110]:


# prediction
y_pred = classifier.predict(x_test)


# In[111]:


acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)

model_results = pd.DataFrame([['Final Logistic Regression', acc, f1, prec, rec]],
                      columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])

results = results.append(model_results, ignore_index=True)
results


# In[112]:


# CROSS VALIDATION
# To decide the final result we are using CV
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
print("Accuracies is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[113]:


# PREDICTING THE SINGLE OBSERVATION


# In[114]:


data.head()


# In[124]:


single_obs = [[17.990000,10.380000,122.800000,1001.000000,0.118400,0.277600,0.300100,0.147100,0.241900,0.078710,1.095000,0.905300,8.589000,153.400000,0.006399,0.049040,0.053730,0.015870,0.030030,0.006193,25.38000,17.330000,184.600000,2019.000000,0.162200,0.665600,0.711900,0.265400,0.460100,0.118900]]


# In[125]:


single_obs


# In[126]:


data.iloc[0]


# In[127]:


classifier.predict(sc.transform(single_obs))


# In[ ]:


# Array[1] means that it is of malignant type
# Therefore First Row Patient information is Malignant

