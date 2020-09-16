from flask import  Flask,request,render_template
from keras.models import  load_model
from keras.preprocessing.image import load_img
import numpy as np
import  os

def preprocessImage(imagepath):
    image=load_img(imagepath)
    image=image.resize((224,224))
    image=np.array(image)
    image=image/255.0
    img=image.reshape(1,224,224,3)
    return img

def get_celebrity(pred):
    if pred==0:
        return "Cristiano Ronaldo"
    elif pred==1:
        return "Michael Phelps"
    elif pred==2:
        return "Roger Federer"
    elif pred==3:
        return "Virat Kohli"

app = Flask(__name__)
model=load_model("finalModel.h5")


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    image=request.files['image']
    filename=image.filename
    file_path=os.path.join('static/images',filename)
    image.save(file_path)

    img=preprocessImage(file_path)
    prediction=np.argmax(model.predict(img))

    output=get_celebrity(prediction)


    return render_template('result.html',user_image=file_path,name=output)
if __name__ == "__main__":
    app.run(debug=True)