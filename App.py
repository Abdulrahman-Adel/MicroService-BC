# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:05:02 2020

@author: Abdelrahman
"""

import numpy as np
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("finalized_model.pkl", "rb"))


df = pd.DataFrame(columns=['variable1', 'variable2', 'variable3', 'variable5', 'variable6',
       'variable7', 'variable8', 'variable9', 'variable10', 'variable11',
       'variable12', 'variable13', 'variable14', 'variable15', 'variable18',
       'variable19'])



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    features = [np.nan if str(i) in ["nan","NAN"] else i for i in request.form.values()]
    df.loc[len(df)] = features
    prediction = model.predict(df)


    return render_template("index.html", prediction_text="Class Label should be {}".format(prediction[-1]))


if __name__ == "__main__":
    app.run(debug=False)

