

# ***About this Notebook***

# <span style="font-family:Montserrat">Real or Not? NLP with Disaster Tweets</span>

#### 

> #### Predict which Tweets are about real disasters and which ones are not ...???
>
> #### This is practice classifier on *[one of Kaggle problems](https://www.kaggle.com/c/nlp-getting-started)* using machine learning methods.



<span style="font-size:24px"> For this competition I used ***Multinomial Naive Bayes Classifier*** and my ***score is 0.78425*** [My Notebook](https://www.kaggle.com/knshruthikompella/kernel26cbbf7b4e)</span>

### Contents :
1. Reading the given csv files
2. Treating the Missing Values
3. Exploring the Target Column
4. Data Preprocessing ( ***Tweets, Wordcount & Wordcloud*** )
    * Data Cleansing
    * Tokenization
    * Stopwords Removal 
5. Building a Text Classification model
6. Submission

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

```
/kaggle/input/nlp-getting-started/sample_submission.csv
/kaggle/input/nlp-getting-started/test.csv
/kaggle/input/nlp-getting-started/train.csv
```

# <span style="color:#008abc">1. Reading the given csv files</span>


```python
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
```

><span style="color:#008abc">**Let's see how train and test data looks like .. **</span>


```python
train_data.head()
#train_data[:10]
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>



```python
test_data.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Heard about #earthquake is different cities, s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>there is a forest fire at spot pond, geese are...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
  </tbody>
</table>


# <span style="color:#008abc">2. Treating the Missing Values</span>

><span style="color:#008abc">**Fill out the missing values with a keyword ..**</span>


```python
train_data.fillna('Unavailable') # fill out the missing values in the train dataset
test_data.fillna('Unavailable') # fill out the missing values in the test dataset
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Unavailable</td>
      <td>Unavailable</td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Unavailable</td>
      <td>Unavailable</td>
      <td>Heard about #earthquake is different cities, s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Unavailable</td>
      <td>Unavailable</td>
      <td>there is a forest fire at spot pond, geese are...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>Unavailable</td>
      <td>Unavailable</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>Unavailable</td>
      <td>Unavailable</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3258</th>
      <td>10861</td>
      <td>Unavailable</td>
      <td>Unavailable</td>
      <td>EARTHQUAKE SAFETY LOS ANGELES ÛÒ SAFETY FASTE...</td>
    </tr>
    <tr>
      <th>3259</th>
      <td>10865</td>
      <td>Unavailable</td>
      <td>Unavailable</td>
      <td>Storm in RI worse than last hurricane. My city...</td>
    </tr>
    <tr>
      <th>3260</th>
      <td>10868</td>
      <td>Unavailable</td>
      <td>Unavailable</td>
      <td>Green Line derailment in Chicago http://t.co/U...</td>
    </tr>
    <tr>
      <th>3261</th>
      <td>10874</td>
      <td>Unavailable</td>
      <td>Unavailable</td>
      <td>MEG issues Hazardous Weather Outlook (HWO) htt...</td>
    </tr>
    <tr>
      <th>3262</th>
      <td>10875</td>
      <td>Unavailable</td>
      <td>Unavailable</td>
      <td>#CityofCalgary has activated its Municipal Eme...</td>
    </tr>
  </tbody>
</table>
<p>3263 rows × 4 columns</p>


><span style="color:#008abc">**I have used Deleting Columns technique. I have just considered text & target columns of train data and id & text columns of test data ..**</span>
>
> Also, Let's see how the data looks like..


```python
train = train_data[['text','target']].copy()
rawTextData = train_data['text'].copy()
print(train.head())
```

                                                    text  target
    0  Our Deeds are the Reason of this #earthquake M...       1
    1             Forest fire near La Ronge Sask. Canada       1
    2  All residents asked to 'shelter in place' are ...       1
    3  13,000 people receive #wildfires evacuation or...       1
    4  Just got sent this photo from Ruby #Alaska as ...       1



```python
test = test_data[['id','text']].copy()
print(test.head())
```

       id                                               text
    0   0                 Just happened a terrible car crash
    1   2  Heard about #earthquake is different cities, s...
    2   3  there is a forest fire at spot pond, geese are...
    3   9           Apocalypse lighting. #Spokane #wildfires
    4  11      Typhoon Soudelor kills 28 in China and Taiwan



```python
print('train.shape ',train.shape)
print('test.shape ', test.shape)
```

    train.shape  (7613, 2)
    test.shape  (3263, 2)


# <span style="color:#008abc">3. Exploring the Target Column</span>


```python
train['target'].value_counts()
```




    0    4342
    1    3271
    Name: target, dtype: int64




```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(train['target'].value_counts().index,train['target'].value_counts(),palette='Accent')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f63af543438>




![Target Data](https://shruthikompella.netlify.com/post/myblog/myblog_14_1.png)

<span style="color:#008abc">**Importing Major Libraries ..**</span>

```python
import logging
from numpy import random
import gensim
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
%matplotlib inline
```

# <span style="color:#008abc">4. Data Preprocessing</span>
>[Reference for Text Preprocessing](https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f) <- Click here
>
><span style="color:#008abc">**Tweets, Wordcount & Wordcloud ..**</span>


```python
# A disaster tweet
disaster_tweets = train[train['target']==1]['text']
print("Disater Tweet  :  ",disaster_tweets[50])
#A Non-disaster tweet
non_disaster_tweets = train[train['target']==0]['text']
print("Non-Disater Tweet  :  ",non_disaster_tweets[20])
```

    Disater Tweet  :   Deputies: Man shot before Brighton home set ablaze http://t.co/gWNRhMSO8k
    Non-Disater Tweet  :   this is ridiculous....



```python
print(train['text'].apply(lambda x: len(x.split(' '))).sum())
print(test['text'].apply(lambda x: len(x.split(' '))).sum())
```

    113654
    48876



```python
from wordcloud import WordCloud
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[28, 15])

wordcloud1 = WordCloud( background_color='white',width=700,height=500).generate(" ".join(disaster_tweets)) # wordcloud for disaster tweets

ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Disaster Tweets',fontsize=30);

wordcloud2 = WordCloud( background_color='white',width=700,height=500).generate(" ".join(non_disaster_tweets)) # wordcloud for non-disaster tweets

ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Non Disaster Tweets',fontsize=30);
```


![WordCloud](https://shruthikompella.netlify.com/post/myblog/myblog_19_0.png)


## ***DATA CLEANSING***
>data cleaning or data cleansing means filtering and modifying your data such that it is easier to explore, understand, and model.
* HTML decoding
* Transform words to lowercase/uppercase
* Remove unwanted words, symbols, numbers
* Remove punctuations
* Deleting the columns with missing values





```python
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    operators_replace = re.compile('[/(){}\[\]\|@,;]')
    symbols_number_replace = re.compile('[^0-9a-z #+_]')
    text = BeautifulSoup(text, "lxml").text 
    text = text.lower() 
    text = operators_replace.sub(' ', text)
    text = symbols_number_replace.sub('', text)
    text = re.sub('\n', '', text)
    return text

# Applying the cleaning function to both test and training datasets
train['text'] = train['text'].apply(clean_text)
test['text'] = test['text'].apply(clean_text)

# Let's take a look at the updated text
train['text'].head()
```




    0    our deeds are the reason of this #earthquake m...
    1                forest fire near la ronge sask canada
    2    all residents asked to shelter in place are be...
    3    13 000 people receive #wildfires evacuation or...
    4    just got sent this photo from ruby #alaska as ...
    Name: text, dtype: object



## ***TOKENIZATION***
>This breaks up the strings into a list of words or pieces based on a specified pattern using Regular Expressions aka RegEx. The pattern I chose to use this time (r'\w') also removes punctuation and is a better option for this data in particular.


```python
# Tokenizing the training and the test set
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
train['text'] = train['text'].apply(tokenizer.tokenize)
test['text'] = test['text'].apply(tokenizer.tokenize)
train['text'].head()
```




    0    [our, deeds, are, the, reason, of, this, earth...
    1        [forest, fire, near, la, ronge, sask, canada]
    2    [all, residents, asked, to, shelter, in, place...
    3    [13, 000, people, receive, wildfires, evacuati...
    4    [just, got, sent, this, photo, from, ruby, ala...
    Name: text, dtype: object



## ***STOPWORDS REMOVAL***
>Imported a list of the most frequently used words from the NL Toolkit at the beginning with from nltk.corpus import stopwords. There are 179 English words, including ‘i’, ‘me’, ‘my’, ‘myself’, ‘we’, ‘you’, ‘he’, ‘his’, for example. We usually want to remove these because they have low predictive power. 
* Remove Stopwords


```python
def stopwords_remove(text):
    """
    Removing stopwords belonging to english language
    
    """
    STOPWORDS = set(stopwords.words('english'))
    text = ' '.join(word for word in text if word not in STOPWORDS)
   
    return text

# Applying the stopwords function to both test and training datasets
train['text'] = train['text'].apply(stopwords_remove)
test['text'] = test['text'].apply(stopwords_remove)
train.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>deeds reason earthquake may allah forgive us</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>forest fire near la ronge sask canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>residents asked shelter place notified officer...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13 000 people receive wildfires evacuation ord...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>got sent photo ruby alaska smoke wildfires pou...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>


## ***COMPARING DATA***
><span style="color:#008abc">**The data on the left is the raw data and the one on the right is the data after cleaning ..**</span>


```python
rawTextData = rawTextData.head(10)
cleanTextData = train["text"].head(10)
frames = [rawTextData, cleanTextData]
result = pd.concat(frames, axis=1, sort=False)
result
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>deeds reason earthquake may allah forgive us</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>forest fire near la ronge sask canada</td>
    </tr>
    <tr>
      <th>2</th>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>residents asked shelter place notified officer...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>13 000 people receive wildfires evacuation ord...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>got sent photo ruby alaska smoke wildfires pou...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>#RockyFire Update =&gt; California Hwy. 20 closed...</td>
      <td>rockyfire update california hwy 20 closed dire...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>#flood #disaster Heavy rain causes flash flood...</td>
      <td>flood disaster heavy rain causes flash floodin...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>I'm on top of the hill and I can see a fire in...</td>
      <td>im top hill see fire woods</td>
    </tr>
    <tr>
      <th>8</th>
      <td>There's an emergency evacuation happening now ...</td>
      <td>theres emergency evacuation happening building...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>I'm afraid that the tornado is coming to our a...</td>
      <td>im afraid tornado coming area</td>
    </tr>
  </tbody>
</table>


> <span style="color:#008abc">**Model_selection** is a method for setting a **blueprint** to analyze data and then using it to measure new data. Selecting a proper model allows you to generate **accurate results** when making a prediction.To do that, you need to **train your model** by using a specific dataset. Then, you test the model against another dataset.If you have **one dataset**, you'll need to split it by using the Sklearn `train_test_split` function first.</span>

`train_test_split` is a function in **Sklearn model selection** for splitting data arrays into **two subsets**: for training data and for testing data. With this function, you don't need to divide the dataset manually.


```python
X = train.text
y = train.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
```

# <span style="color:#008abc">5. Naive Bayes ( Building a Text Classification model )</span>

[Reference for building a classifier](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) <- Click here

* Accuracy Score
* Classification Report


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

mnb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
mnb.fit(X_train, y_train)

%time
from sklearn.metrics import classification_report
y_pred = mnb.predict(X_test)

print(y_pred)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
```

    CPU times: user 3 µs, sys: 0 ns, total: 3 µs
    Wall time: 7.39 µs
    [0 0 1 ... 1 1 0]
    accuracy 0.8038528896672504
                  precision    recall  f1-score   support
    
               0       0.78      0.91      0.84      1318
               1       0.84      0.66      0.74       966
    
        accuracy                           0.80      2284
       macro avg       0.81      0.78      0.79      2284
    weighted avg       0.81      0.80      0.80      2284


​    

><span style="color:#008abc">**Target Column of Test Data ..**</span>


```python
sub_pred = mnb.predict(test.text)
sub_df = pd.DataFrame({'id':test.id, 'target':sub_pred})
print(sub_df.target.value_counts())
```

    0    2206
    1    1057
    Name: target, dtype: int64


# <span style="color:#008abc">6. Submission</span>


```python
sub_df.to_csv('submission.csv', index=False)
output = pd.read_csv('submission.csv')
output
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3258</th>
      <td>10861</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3259</th>
      <td>10865</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3260</th>
      <td>10868</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3261</th>
      <td>10874</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3262</th>
      <td>10875</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3263 rows × 2 columns</p>



