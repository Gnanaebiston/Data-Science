from textblob import TextBlob
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

def get_sentiments(polarity):
	if polarity>0:
		return 'Positive'
	elif polarity<0:
		return 'Negative'
	else:
		return 'Neutral'

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
	tweet = request.form["tweet"]
	object=TextBlob(tweet)
	polarity=object.sentiment.polarity
	senti=get_sentiments(polarity)
	return render_template('index.html',prediction_text=senti)
if __name__ == "__main__":
    app.run(debug=True)