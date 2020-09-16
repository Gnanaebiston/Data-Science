from flask import Flask,request,render_template
import pickle
import numpy as np
from keras.models import load_model

def fuel_type(fueltype):
    if fueltype=='Diesel':
        return 3
    elif fueltype=='Petrol':
        return 2
    elif fueltype=='CNG':
        return 1;
def transmissionn(transmission):
    if transmission=='Automatic':
        return 1
    elif transmission=='Manual':
        return 2

def OwnerType(ownertype):
    if ownertype=="First":
        return 3
    elif ownertype=="Second":
        return 2
    elif ownertype=="Third":
        return 1

def predict_price(l):
    temp = scaler.transform([l])
    return round(regressor.predict(temp)[0],3)



app = Flask(__name__)
regressor=pickle.load(open('regressor.pkl','rb'))
scaler=pickle.load(open("scaler.pkl",'rb'))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    builtyear=request.form['builtyear']
    kmdriven=request.form['kmdriven']
    fueltype=request.form['fueltype']
    fuel=fuel_type(fueltype)
    transmission=request.form['transmission']
    trans=transmissionn(transmission)
    ownertype = request.form['ownertype']
    owner=OwnerType(ownertype)
    engine = request.form['engine']
    newprice = request.form['newprice']
    lis=[int(builtyear),int(kmdriven),fuel,trans,owner,int(engine),int(newprice)]

    price=predict_price(lis)

    return render_template('index.html',prediction_text=price)
if __name__ == "__main__":
    app.run(debug=True)