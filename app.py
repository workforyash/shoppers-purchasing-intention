from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

#importing model
rf = pickle.load(open("model/randomForest.pkl", 'rb'))

@app.route("/predict", methods=["Get", "POST"])
def predict():
    cols = [
    "PageValues",
    "ExitRates",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "VisitorType_New_Visitor",
    "VisitorType_Returning_Visitor",
    "VisitorType_Other",
    "Month_Aug",
    "Month_Dec",
    "Month_Feb",
    "Month_Jul",
    "Month_June",
    "Month_Mar",
    "Month_May",
    "Month_Nov",
    "Month_Oct",
    "Month_Sep",]

    if request.method=='POST':
        PageValues = float(request.form.get('PageValues'))
        ExitRates = float(request.form.get('ExitValues'))
        ProductRelated = int(request.form.get('ProductRelated'))
        ProductRelated_Duration = float(request.form.get('ProductRelated_Duration'))
        BounceRates = float(request.form.get('BounceRates'))
        VisitorType = request.form.get('VisitorType')
        Month = request.form.get('Month')   

        visitor_type_index = cols.index(VisitorType)
        month_index = cols.index(Month)
        x = np.zeros(len(cols))
        x[0] = PageValues
        x[1] = ExitRates
        x[2] = ProductRelated
        x[3] = ProductRelated_Duration
        x[4] = BounceRates
        x[visitor_type_index] = 1
        x[month_index] = 1

        result = rf.predict([x])[0]

        if result == 1:
            ans = "The shopper's intention is to purchase."
        else:
            ans = "The shopper does not intent to purchase."
        return render_template('home.html', result=ans)
    
    else:
        return render_template('home.html')

@app.route("/")
def hello_world():
    return render_template('index.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
