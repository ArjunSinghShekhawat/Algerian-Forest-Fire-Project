from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


with open("D:\Algerian Forest Fire Dataset Prediction\Algerian_dataset_Prediction\model\obj_regrassor.pkl",'rb') as file:
    ridge = pickle.load(file)
with open("D:\Algerian Forest Fire Dataset Prediction\Algerian_dataset_Prediction\model\scaler.pkl",'rb') as file:
    scaler = pickle.load(file)


@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/predictdata',methods=['POST','GET'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = request.form.get('Temperature')
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        result = ridge.predict(scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]))

        return render_template('result.html',result=result)
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


