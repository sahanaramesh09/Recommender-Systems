#!/usr/bin/env python
# coding: utf-8
####Student Id: 013832065
####Student Name: Sahana Ramesh
####Reference: https://github.com/srafay/Hadoop-hands-on

# In[33]:


#Setup: create a spark session
from __future__ import print_function

import sys
from pyspark.sql.session import SparkSession
spark = SparkSession        .builder        .appName("AverageMovieRating")        .getOrCreate()


# In[34]:


#read the data set : ratings_small.csv which contains 4 columns userId,movieId,rating,timestamp
dataset = spark.read.csv('ratings.csv')


# In[35]:


#Extract just the movieId and rating columns from the dataset and store it in required variable
required = dataset.rdd.map(lambda r: (r[1],r[2]))


# In[36]:


# use the collect() function to see what the movieRatings variable contains
required.collect()


# In[37]:


#First() function gives us the first row 
header = required.first()
#We want to remove the header to carry out numerical calculations on the dataset
intermediate=required.filter(lambda mydata: mydata != header)
intermediate.collect()


# In[38]:


#intermediate data contains string values of movieId and ratings. So we have to convert them to numbers to carry out calculations.
movieRatings = intermediate.map(lambda r: (int(r[0]),float(r[1])))
movieRatings.collect()


# In[39]:


#Here we are creting a tuple of (movieId,(ratings,1)) bu using the map function. 1 is added in the tuple to get a count of the total ratings.
movieRatingsTuple = movieRatings.map(lambda r: (r[0],(r[1],1)))


# In[40]:


movieRatingsTuple.collect()


# In[41]:


#Here we are use the reduceByKey function where the ratings and count get added up based on key i.e movieId
ratingsSumCount = movieRatingsTuple.reduceByKey(lambda movie1, movie2: ( movie1[0] + movie2[0], movie1[1] + movie2[1] ) )


# In[42]:


ratingsSumCount.collect()


# In[43]:


#Here we are dividing the sum of ratings by count to get averageRatings
averageRatings = ratingsSumCount.mapValues(lambda totalAndCount : totalAndCount[0] / totalAndCount[1])


# In[44]:


averageRatings.collect()


# In[45]:


averageRatings.coalesce(1).saveAsTextFile('013832065_report')


# In[ ]:




