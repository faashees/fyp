#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install snscrape')


# In[56]:


import pandas as pd
import snscrape.modules.twitter as sntwitter
from nltk.corpus import stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer
import re


# In[94]:


#search term

query = "('oneXOX  murah OR oneXOX laju OR oneXOX slow OR oneXOX  bagus OR oneXOX mahal OR oneXOX  tiada line' )  until:2023-03-31 since:2019-03-31"

tweets = []
limit = 150


for tweet in sntwitter.TwitterHashtagScraper(query).get_items():
    
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.url, tweet.user.username, tweet.sourceLabel, tweet.user.location, tweet.content])
        
df = pd.DataFrame(tweets, columns=['Date', 'TweetURL','User', 'Source', 'Location', 'Tweet'])

df.to_csv('telcooneXOX.csv')


# In[95]:


df.head()


# In[96]:


df


# In[85]:


df.Tweet.str.split()


# In[76]:


df.Tweet.str.split().to_csv('telcoYes.csv') #write dataframe into csv file
savedTweets = pd.read_csv('telcoYes.csv',index_col=0) #reads csv file


# In[47]:


flat_list = sum(splitted, [])


# In[48]:


len(flat_list)


# In[62]:


flat_list


# In[49]:


flat_list = [word.lower() for word in flat_list]


# In[50]:


stops = (stopwords.words('english'))


# In[51]:


stops.append ('&amp;')


# In[15]:


flat_list = [word for word in flat_list if word not in stops]


# In[16]:


arr = pd.Series(flat_list)


# In[61]:


arr.value_counts().head()


# In[87]:


df.describe()


# In[ ]:




