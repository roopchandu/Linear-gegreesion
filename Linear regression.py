#!/usr/bin/env python
# coding: utf-8

# In[1]:


#I have taken the college dataset to predict the chances of getting the admission in the university with the help of linear regression.
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)


# In[2]:


df = pd.read_csv("c:/Users/ROOPCHAND PARISE/Downloads/college1234.csv")


# In[4]:


df.head()


# In[5]:


df.head(7)


# In[15]:


#Boxplot helps to findout the outliers.
df.plot(kind='box', subplots=True, layout=(2,5), sharex=False, sharey=False)
plt.tight_layout()
plt.show()


# In[16]:


#Serial number in the dataset doesnot have any impact so iam dropping this variable 
df.drop(labels='Serial No.', axis=1, inplace=True) 
df.columns = df.columns.str.replace(' ', '_')


# In[17]:


print(df.columns)


# In[7]:


df.describe()


# In[24]:


df.info()


# In[25]:


#Exploratory data analysis
sns.pairplot(df)


# In[26]:


#correlation matrix
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')


# In[52]:


#MULTIPLE LINEAR REGRESSION
# As from the above graph it is clear that TOEFL Score do not have a strong relationship with the Chance of Admit as compared with rest 
# two variables.
# Lets build 2 algorithm with and without TOEFL Score to get more clear picture.
import statsmodels.formula.api as sm

model1 = sm.ols(formula='Chance_of_Admit_ ~ GRE_Score + CGPA + TOEFL_Score ', data= df).fit()
model2 = sm.ols(formula='Chance_of_Admit_ ~GRE_Score+ CGPA ', data=df).fit()


# In[56]:


lm1 = sm.ols(formula='Chance_of_Admit_~GRE_Score+TOEFL_Score+University_Rating+SOP+LOR_+CGPA+Research', data = df).fit()
lm1.summary()
# 0.805


# In[43]:


lm13 = sm.ols(formula='Chance_of_Admit_~GRE_Score+TOEFL_Score+LOR_+CGPA+Research', data = df).fit()
lm13.summary()
# 0.803


# In[45]:


#SIMPLE LINEAR REGRESSION.
slr3 = sm.ols(formula='Chance_of_Admit_ ~LOR_', data=df).fit()
slr3.summary()
# 0.449


# In[46]:


slr4 = sm.ols(formula='Chance_of_Admit_ ~Research', data=df).fit()
slr4.summary()
# 0.306


# In[47]:


slr5 = sm.ols(formula='Chance_of_Admit_ ~TOEFL_Score', data=df).fit()
slr5.summary()
# 0.627


# In[48]:


slr6 = sm.ols(formula='Chance_of_Admit_ ~GRE_Score', data=df).fit()
slr6.summary()
# 0.644


# In[49]:


slr6 = sm.ols(formula='Chance_of_Admit_ ~GRE_Score', data=df).fit()
slr6.summary()
# 0.644

