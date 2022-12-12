## Loading necessary libraries
import nlp_utils
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
df=pd.read_csv('train.csv')
pd.set_option('display.max_colwidth', -1)
df=df.dropna()
df.reset_index(inplace=True)
import re
import string
# remove all numbers with letters attached to them
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)

# .lower() - convert all strings to lowercase
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

# Remove all '\n' in the string and replace it with a space
remove_n = lambda x: re.sub("\n", " ", x)

# Remove all non-ascii characters
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)

# Apply all the lambda functions wrote previously through .map on the comments column
df['text'] = df['text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk

nltk.download('stopwords')
ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

Y=df['label']
## We select the label column as Y


X_train, X_test, Y_train, Y_test = train_test_split(df['text'], Y, test_size=0.30, random_state=40)
## We have split the data into 70 percent train and 30 percent test

#Applying tfidf to the data set
#tfidf_vect = TfidfVectorizer(stop_words = 'english',max_df=0.7)


count_vect = CountVectorizer(stop_words = 'english')
count_train = count_vect.fit_transform(X_train.values)
count_test = count_vect.transform(X_test.values)

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier().fit(count_train,Y_train)
#predict on train
train_preds3 = RF.predict(count_train)
#accuracy on train

#predict on test
test_preds3 = RF.predict(count_test)
from sklearn.externals import joblib

joblib.dump(RF, 'model.pkl')
