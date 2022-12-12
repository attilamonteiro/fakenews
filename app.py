from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

df=pd.read_csv('train.csv')
count_vect = CountVectorizer(stop_words = 'english')
app = Flask(__name__)
Y=df['label']
X_train, X_test, Y_train, Y_test = train_test_split(df['text'], Y, test_size=0.2, random_state=0)

def fake_news_det(news):
    count_train = count_vect.fit_transform(X_train.values.astype('U'))
    count_test = count_vect.transform(X_test.values.astype('U'))
    input_data = [news]
    loaded_model = joblib.load(open('model.pkl', 'rb'))
    vectorized_input_data = count_vect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)