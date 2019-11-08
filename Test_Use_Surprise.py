#!/usr/bin/env python
# coding: utf-8

# In[1]:


import surprise


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv("training.csv")


# In[4]:


data.head()


# In[5]:


data_training = data.drop(['helpful', 'reviewText', 'reviewerName', 'reviewTime', 'summary','unixReviewTime'], axis=1)


# In[6]:


data_training.head()


# In[7]:


lower_rating = data_training['overall'].min()
upper_rating = data_training['overall'].max()
print('Review range: {0} to {1}'.format(lower_rating, upper_rating))


# In[8]:


data_training.overall.value_counts()


# In[9]:


data_training.info()


# In[10]:


data_training.isnull().sum()


# In[11]:


data_training = data_training[['reviewerID','asin','overall']]


# In[12]:


data_training.head()


# In[13]:


from surprise import Reader, Dataset
reader = Reader()
data_surprise = Dataset.load_from_df(data_training[['reviewerID', 'asin', 'overall']], reader)


# In[14]:


data_surprise


# In[33]:


#from surprise.model_selection import train_test_split

#trainset, testset = train_test_split(data_surprise, test_size=0.15)
trainset_full = data_surprise.build_full_trainset()


# In[16]:


#testset[0]


# In[34]:


from surprise import SVD, accuracy
algo = SVD()
algo.fit(trainset_full)


# In[20]:


predictions = algo.test(testset)


# In[21]:


from surprise import accuracy
accuracy.rmse(predictions)


# In[ ]:





# In[22]:


data_test = pd.read_csv("test_with_asin_reviewerID.csv")


# In[23]:


data_test.head()


# In[24]:


data_test['overall'] = 0


# In[25]:


data_test.head()


# In[26]:


data_surprise_test = Dataset.load_from_df(data_test, reader)


# In[27]:


trainset2, testset2 = train_test_split(data_surprise_test, test_size=0.90)


# In[57]:


estimation = []
for i in range(len(data_test)) :
    p = data_test.loc[i,"reviewerID"]
    q = data_test.loc[i,"asin"]
    r = data_test.loc[i,"overall"]
    estimation.append(algo.predict(p,q,r))
    #print(p,q,r,algo.predict(p,q,r))


# In[58]:


estimation


# In[114]:


result = pd.DataFrame(list(estimation))


# In[115]:


result.head()


# In[116]:


result = result.drop(['r_ui', 'details'], axis=1)


# In[117]:


result.head()


# In[118]:


result["key"] = result["uid"].astype(str) + "-" + result["iid"].astype(str)


# In[119]:


result.head()


# In[120]:


result_final = result[['key','est']]


# In[121]:


result_final.head()


# In[122]:


result_final = result_final.rename(columns = {"est": "overall"}) 


# In[123]:


result_final.dtypes


# In[124]:


result_final.head()


# In[125]:


result_final.count()


# In[126]:


result_final.to_csv('013832065_Submission.csv', index=False)


# In[25]:


from surprise import KNNWithMeans


# In[28]:


algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo.fit(trainset)


# In[37]:


train_pred = algo.test(testset)


# In[38]:


accuracy.rmse(train_pred)


# In[ ]:




