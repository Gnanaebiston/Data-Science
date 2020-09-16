import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle,re,nltk
import tensorflow
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def cleaner(text,classifier,cv):
    stemmer=PorterStemmer()
    sentence = []
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9 ]', ' ', text)
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sent = ' '.join(words)
    temp = cv.transform([sent])
    answer = classifier.predict(temp)
    return answer

def get_class(label):
    if label==1:
        return  "Spam"
    else:
        return "Ham"


app = Flask(__name__)
model=pickle.load(open("model.pkl",'rb'))
cv=pickle.load(open("cv.pkl",'rb'))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    data=request.form['tweet']
    output=cleaner(data,model,cv)
    label_name=get_class(output)
    return render_template('index.html',prediction_text=label_name)
if __name__ == "__main__":
    app.run(debug=True)