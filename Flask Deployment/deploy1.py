import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
app = Flask(__name__)


'''   'processor_brand', 'processor_gnrtn', 'graphic_card_gb', 'weight',
       'warranty', 'msoffice', 'Pentium_Quad', 'Celeron_Dual', 'Core',
       'Ryzen'   '''


'''   'processor_brand', 'processor_gnrtn', 'graphic_card_gb', 'Pentium_Quad', 'Celeron_Dual', 'Core',
       'Ryzen'   '''

#i will load only the features that will be used by my model

warranty_encoded = pickle.load(open('warranty_encoder.pkl', 'rb'))
msoffice_encoded = pickle.load(open('msoffice_encoder.pkl', 'rb'))
weight_encoded = pickle.load(open('weight_encoder.pkl', 'rb'))

scaler = pickle.load(open('deploymentScaler.pkl', 'rb'))
linear = pickle.load(open('linear_model.pkl', 'rb'))


def encode_processor_name(name):
    # Example mapping based on hypothetical target variable (e.g., performance score)
    encoding = {
        'Intel': 10,
        'M1': 1,
        'AMD': 4,
    }
    return encoding.get(name, 0)  # Default to 0 if processor name not found


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Retrieve form inputs
    processor_brand = request.form['processor_brand']
    gnrtn = request.form['processor_gnrtn']
    graphic_card_gb = request.form['graphic_card_gb']
    warranty = request.form['warranty']
    weight = request.form['weight']
    msoffice = request.form['msoffice']
    Pentium_Quad = request.form['Pentium_Quad']
    Celeron_Dual = request.form['Celeron_Dual']
    Core = request.form['Core']
    Ryzen = request.form['Ryzen']



    processor_brand_encoded = encode_processor_name(processor_brand)
    gnrtn_encode = gnrtn.rstrip('th')
    warranty_encode = warranty_encoded.transform([warranty])[0]
    weight_encode = weight_encoded.transform([weight])[0]
    msoffice_encode = msoffice_encoded.transform([msoffice])[0]



    features = np.array([[processor_brand_encoded, gnrtn_encode, graphic_card_gb, warranty_encode,
                          weight_encode, msoffice_encode, Pentium_Quad,Celeron_Dual, Core, Ryzen]])


    # Apply normalization to the features
    final_features = scaler.transform(features)

    # Make a prediction
    prediction = linear.predict(final_features)

    # Render the result to HTML
    return render_template('index.html', prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True)
