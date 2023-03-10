# -*- coding: utf-8 -*-
"""Fake_News_Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1THLQzeAnWHU8HXGyTm17-GprQapCt7zU

# TfidfVectorizer Explanation
Convert a collection of raw documents to a matrix of TF-IDF features

TF-IDF where TF means term frequency, and IDF means Inverse Document frequency.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
text = ['Hello Kushal Bhavsar here, I love machine learning','Welcome to the Machine learning hub' ]

vect = TfidfVectorizer()

vect.fit(text)

## TF will count the frequency of word in each document. and IDF 
print(vect.idf_)

print(vect.vocabulary_)

"""### A words which is present in all the data, it will have low IDF value. With this unique words will be highlighted using the Max IDF values."""

example = text[0]
example

example = vect.transform([example])
print(example.toarray())

"""### Here, 0 is present in the which indexed word, which is not available in given sentence.

## PassiveAggressiveClassifier

### Passive: if correct classification, keep the model; Aggressive: if incorrect classification, update to adjust to this misclassified example.

Passive-Aggressive algorithms are generally used for large-scale learning. It is one of the few ‘online-learning algorithms‘. In online machine learning algorithms, the input data comes in sequential order and the machine learning model is updated step-by-step, as opposed to batch learning, where the entire training dataset is used at once. This is very useful in situations where there is a huge amount of data and it is computationally infeasible to train the entire dataset because of the sheer size of the data. We can simply say that an online-learning algorithm will get a training example, update the classifier, and then throw away the example.

## Let's start the work
"""

#import os
#os.chdir("D:/Fake News Detection")

import pandas as pd

dataframe = pd.read_csv('news.csv')
dataframe.head()

x = dataframe['text']
y = dataframe['label']

x

y

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
y_train

y_train

tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)

"""* max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
* max_df = 25 means "ignore terms that appear in more than 25 documents".
"""

classifier = PassiveAggressiveClassifier(max_iter=50)

.fit(tfid_x_train,y_train)

y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

cf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(cf)

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)

fake_news_det('U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.')

fake_news_det("""Go to Article 
President Barack Obama has been campaigning hard for the woman who is supposedly going to extend his legacy four more years. The only problem with stumping for Hillary Clinton, however, is sheâ€™s not exactly a candidate easy to get too enthused about.  """)

import pickle
pickle.dump(classifier,open('model.pickle', 'wb'))

# load the model from disk
loaded_model = pickle.load(open('model.pickle', 'rb'))
def fake_news_det1(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    print(prediction)

fake_news_det1("""Go to Article 
President Barack Obama has been campaigning hard for the woman who is supposedly going to extend his legacy four more years. The only problem with stumping for Hillary Clinton, however, is sheâ€™s not exactly a candidate easy to get too enthused about.  """)

fake_news_det1("""U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.""")

fake_news_det('''U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.''')

