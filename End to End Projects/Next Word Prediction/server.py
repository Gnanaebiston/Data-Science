from flask import Flask,request,render_template
import pickle
import numpy as np
from keras.models import load_model


def predict_next_word(model, tokenizer, text, n_words):
    result = ''
    out_word = []

    for i in range(n_words):
        input_data = text.split()
        encoded_data = [tokenizer.texts_to_sequences([x])[0] for x in input_data]
        encoded_data = np.array(encoded_data)
        encoded_data = encoded_data.reshape(1, 3)

        output = np.argmax(model.predict(encoded_data))

        for index, word in tokenizer.word_index.items():
            if word == output:
                out_word.append(index)
                text = input_data[1] + " " + input_data[2] + " " + index
    result = ' '.join(out_word)
    return result

app = Flask(__name__)
model=load_model("model.h5")
tokenizer=pickle.load(open("tokenizer.pkl",'rb'))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    data=request.form['word']
    n_word=request.form['points']
    n_word=int(n_word)
    output=predict_next_word(model,tokenizer,data,n_word)

    return render_template('index.html',prediction_text=output)
if __name__ == "__main__":
    app.run(debug=True)