#!/usr/bin/env python
# coding: utf-8

# In[225]:


get_ipython().system('pip install imblearn')
get_ipython().system('pip install mlxtend')


# In[232]:


# Libraries for data preparation & visualisation


import re
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import defaultdict
from numbers import Integral
import itertools
import array


# Library to ignore warnings

import warnings
warnings.filterwarnings("ignore")

# Library for assigning sentiment label

from textblob import TextBlob

# Libraries for text analytics

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter  import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Libraries for ML modeling 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.svm import LinearSVC
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import accuracy_score
from ast import literal_eval
from PIL import Image
from sklearn.utils import shuffle
from array import array


# In[52]:


# Reading in data 

df2 = pd.read_csv('Celcom.csv')
df2.head()


# In[53]:


# Shape of data

df2.shape


# In[54]:


df2.info()


# In[55]:


#df2(['Unnamed: 0', 'Date', 'Location', 'Text'])


#drop any column that contains "Unnamed" in column name
df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]


# In[56]:


df2.info()


# In[57]:


df2


# In[58]:


# Replacing missing records under location with 'unknown'

df2 = df2.fillna('Unknown')


# In[59]:


df2.describe(include='all').T


# In[60]:


df2


# In[61]:


# Checking duplicates 

df2.duplicated().sum()


# In[62]:


# Checking for missing values 

df2.isnull().sum().sum()


# In[63]:


df2


# In[64]:


## NO MISSING VALUE ##
# Feature Engineering & EDA #


# In[71]:


#Create new features from Date

df2['Date'] = pd.to_datetime(df2['Date'])
# Day of the week
df2['day_of_week'] = df2['Date'].dt.dayofweek


# In[73]:


df2['day_of_week'] = df2['day_of_week'].replace({0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday',                                               4:'Friday', 5:'Saturday',6:'Sunday'})


# In[74]:


# Hour 

df2['Hour']= df2['Date'].dt.hour + df2['Date'].dt.minute/60


# In[75]:


df2


# In[37]:


# Filter the dataset to only contain english texts

#df2.drop('NegativeWord', axis=1, inplace=True)
#df2.drop('PositiveWord', axis=1, inplace=True)


# In[76]:


# Value counts for day_of_week

df2['day_of_week'].value_counts()


# In[77]:


df2


# In[78]:


# Let's create target column, i.e., sentiment associated with text

# Defining a function to assign sentiments (positive, negative or neutral)

def get_Sentiment(Text):
    blob = TextBlob(Text)
    Sentiment = blob.sentiment.polarity
    if Sentiment > 0:
        return 'positive'
    elif Sentiment < 0:
        return 'negative'
    else:
        return 'neutral'
    
df2['Sentiment'] = df2['Text'].apply(get_Sentiment)


# In[79]:


# Value counts for sentiment

df2['Sentiment'].value_counts()


# In[80]:


# function to create labeled barplots


def labeled_barplot(df2, feature, target ,perc=False, n=None):

    total = len(df2[feature])  # length of the column
    count = df2[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 5))
    else:
        plt.figure(figsize=(n + 2, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(data=df2.sort_values(by=target),x=feature,palette="Paired",hue=target,                       order=df2[feature].value_counts().index[:n].sort_values(),)

    
    
    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(100 * p.get_height() / total)  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(label,(x, y),ha="center",va="center",size=12,xytext=(0, 5),textcoords="offset points",)  # annotate the percentage

    plt.show()  # show the plot


# In[82]:


# day_of_week

#df2 = df2.copy() 

from pandas.api.types import CategoricalDtype
cats     = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
cat_type = CategoricalDtype(categories=cats, ordered=True)

df2['day_of_week']=df2['day_of_week'].astype(cat_type)

labeled_barplot(df2, 'day_of_week', 'Sentiment')


# In[83]:


# hour

plt.figure(figsize=(5,5))
sns.histplot(data=df2, x='Hour');


# In[84]:


sns.set_theme(style="darkgrid")

sns.lineplot(x="day_of_week", y="Hour",  data=df2)


# In[86]:


#df2.drop('Date', axis=1, inplace=True)


# In[88]:


#df2.drop('User', axis=1, inplace=True)


# In[135]:


#df2.drop('Location', axis=1, inplace=True)


# In[134]:


df2


# In[251]:


#####################################################################################################################################


# In[90]:


positive_words = pd.read_csv('Positive.txt', skiprows=35, names=['words'])
positive_words = positive_words['words'].values.tolist()


# In[136]:


negative_words = pd.read_csv('Negative.txt', skiprows=35, names=['words'])
negative_words = negative_words['words'].values.tolist()


# In[137]:


# Let's create new features, i.e., number of positive & negative words associated with each text

# Defining a function to count number of positive & negative words

def count_words(tweet, words): 
    count = 0
    for word in tweet.split(' '):
        if word in words:
            count += 1
    return count


# In[138]:



df2['PositiveWord'] = df2['Text'].apply(count_words, words=positive_words)
df2['NegativeWord'] = df2['Text'].apply(count_words, words=negative_words)


# In[139]:


# Function to create stacked barplots

def stacked_barplot(df2, predictor, target):

    count = df2[predictor].nunique()
    sorter = df2[target].value_counts().index[-1]
    tab1 = pd.crosstab(df2[predictor], df2[target], margins=True).sort_values(by=sorter, ascending=False)
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(df2[predictor], df2[target], normalize="index").sort_values(by=predictor, ascending=True)
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(loc="lower left", frameon=False,)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# In[140]:


df2.head()


# In[141]:


# number_positive_words

stacked_barplot(df2,'PositiveWord', 'Sentiment')


# In[142]:


#Tweets with 0 or 1 count for number_positive_words have all three sentiments 'neutral', 'positive' & 'negative'
#Tweets with 1, 2, 3 or 4 number_positive_words are majorly a 'positive' sentiment as expected


# In[143]:


# number_positive_words

stacked_barplot(df2, 'NegativeWord', 'Sentiment')


# In[144]:


#Tweets with 0 or 1 count for number_negative_words have all three sentiments 'neutral', 'positive' & 'negative'
#Tweets with 2, 3 or 4 number_negative_words are majorly a 'negative' sentiment as expected


# In[145]:


df2.head()


# In[146]:


#df2.to_csv('SentimentText.csv')


# In[186]:


# Convert tweets to lowercase 

df2['Text'] = df2['Text'].str.lower()

# Remove non-alphanumeric character from tweets such as '@'

def remove_non_alphanumeric(tweet):
    pattern = re.compile('\W')
    x = re.sub(pattern, ' ', tweet)
    return(x)
    
df2['Text'] = df2['Text'].apply(remove_non_alphanumeric)

df2.head()


# In[187]:


#The text has been converted to lowercase & non-alphanumerical characters such as @ & # have been removed


# In[188]:


# Tokenize sentences to arrays of words

df2['tokens'] = df2['Text'].apply(nltk.word_tokenize)

df2.head()


# In[189]:


# Lemmatize the words

lem = WordNetLemmatizer()
df2['tokens'] = df2['tokens'].apply(lambda x: [lem.lemmatize(word) for word in x])

df2.head()


# In[190]:


# Stemming the words

stem = PorterStemmer()
df2['tokens'] = df2['tokens'].apply(lambda x: [stem.stem(word) for word in x])

df2['tokens'] = df2['tokens'].str.join(' ')

df2.head()


# In[191]:


# Let's plot a word cloud to see the difference between original tweets & tweets post data cleaning

# Original tweets

df2['Text'].to_csv('Tweetsonly.csv')  
text = open('tweetsonly.csv').read()

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[153]:


# tweets post data cleaning

df2['tokens'].to_csv('cleanedtweetsonly.csv')  
text2 = open('cleanedtweetsonly.csv').read()

wordcloud = WordCloud().generate(text2)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[154]:


# Clean up locations column

#df2['Location'].value_counts().head(150)


# In[155]:


#df2 = df2.drop('User', axis=1)
#df2


# In[207]:


# +++++++++++++++++++++++++++++++MACHINE LEARNING +++++++++++++++++++++++++++++++#


# In[157]:


df2['Sentiment'] = df2['Sentiment'].replace({'neutral':0, 'positive':1, 'negative':-1})
df2['Sentiment'] = df2['Sentiment'].astype(int)


# In[158]:


# Assign features & target as X & y respectively

X = df2.drop('Sentiment', axis=1)
y = df2['Sentiment']


# In[159]:


# One hot encoding of day_of_week column

X = pd.get_dummies(X, columns=['day_of_week'], drop_first=True)


# In[160]:


# Split data into train & test set using stratify to maintain the split of sentiment across train & test sets

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y,                                                   shuffle=True)


# In[161]:


# Shape of splits 

X_train_full.shape


# In[162]:


X_test.shape


# In[163]:


y_train_full.value_counts(normalize=True)


# In[164]:


y_test.value_counts(normalize=True)


# In[165]:


#++++++++++++++++ Stratify has maintained the split of #
#neutral ('0'), positive ('1') & negative ('-1')#
#tweets in the dataset +++++++++++++++++++++++++++++++ #


# In[166]:


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++ 5-fold cross validation scores (default model parameters) ++++++++++++++
# ++++++++++++++ We would like each class to be predicted correctly. ++++++++++++++++++++
# ++++++++++++++ Tuning the 'F1' metric will ensure the maximum possibility +++++++++++++
# ++++++++++++++ of correct predictions across each of the target classes +++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# In[167]:


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Further, the following additional points need consideration +++++++++++++++++++++++++++++++++++
# 1. Vectorizer 'TFIDF' is chosen to convert text data to numerical feature matrix ++++++++++++++
# 2. RandomOverSampler would be needed to handle class imbalance across target classes++++++++++++
# 3. Further, choice of ML model(s) that work well with textual data +++++++++++++++++++++++++++++
# Multinomial Naive Baiyes, Linear Support Vector Classifier, Random Forest Classifier & XGBoost+++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# In[192]:


df2


# In[193]:


df2.info()


# In[214]:


# Initialize vectorizer TfidfVectorizer with default parameters, & stop_words as english

vectorizer  = TfidfVectorizer(stop_words='english')       # using TfIdf to make words as features by making word vectors
x= vectorizer.fit_transform(df2['Text'])
y= df2.Sentiment


# In[215]:


x_train,x_test,y_train,y_test= train_test_split(x,y,random_state= 42)         # splitting data for cross validation


# In[216]:


from sklearn.linear_model import LogisticRegressionCV      # using multiNomial Naive Bayes as classifier

clf= LogisticRegressionCV()

clf.fit(x_train,y_train)
pred= clf.predict(x_test)


# In[217]:


clf.Cs                          #best C parameter


# In[218]:


from sklearn.metrics import classification_report , accuracy_score
print('accuracy=',accuracy_score(y_test,pred))
print(classification_report(y_test, pred))


# In[228]:



CM = confusion_matrix(y_test, pred)
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(10, 5))
plt.show()


# In[229]:


def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes

importance = get_most_important_features(vectorizer, clf, 30)


# In[231]:


def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    
    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    
    fig = plt.figure(figsize=(80, 80))  

    plt.subplot(121)
    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
    plt.title('Negative Review', fontsize=40)
    plt.yticks(y_pos, bottom_words, fontsize=40)
    plt.suptitle('Key words', fontsize=40)
    plt.xlabel('Importance', fontsize=40)
    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Positive Review', fontsize=40)
    plt.yticks(y_pos, top_words, fontsize=40)
    plt.suptitle(name, fontsize=40)
    plt.xlabel('Importance', fontsize=40)
    
    plt.subplots_adjust(wspace=1.8)
    plt.show()

top_scores = [a[0] for a in importance[0]['tops']]
top_words = [a[1] for a in importance[0]['tops']]
bottom_scores = [a[0] for a in importance[0]['bottom']]
bottom_words = [a[1] for a in importance[0]['bottom']]

plot_important_words(top_scores, top_words, bottom_scores, bottom_words, 
                    "Most important words for relevance\n(Lemmatized)")


# In[233]:



clf=MultinomialNB()
clf.fit(x_train,y_train)
pred= clf.predict(x_test)
from sklearn.metrics import classification_report , accuracy_score
print('accuracy=',accuracy_score(y_test,pred))
print(classification_report(y_test, pred))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, pred)
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(10, 5))
plt.show()


# In[234]:


clf=BernoulliNB()
clf.fit(x_train,y_train)
pred= clf.predict(x_test)
from sklearn.metrics import classification_report , accuracy_score
print('accuracy=',accuracy_score(y_test,pred))
print(classification_report(y_test, pred))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, pred)
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(10, 5))
plt.show()


# In[ ]:




