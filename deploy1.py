import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
app = Flask(__name__)


def convert_to_float(gnrtn):
    if pd.isnull(gnrtn):
        return gnrtn
    else:
        return int(gnrtn.rstrip('th'))

#coreencoder = pickle.load(open('core_encoder.pkl', 'rb'))
# Load the function from a pickle file
convert_to_float = pickle.load(open('processor_gnrtn_convert_to_float.pkl','rb'))
linear = pickle.load(open('linear.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    # Retrieve form inputs
    spec = request.form['processor gnrtn']

    conversion = convert_to_float(spec)
    # Combine features into a single array
    features = np.array([[conversion]])

    # Apply normalization to the features
    final_features = scaler.transform(features)

    # Make a prediction
    prediction = linear.predict(final_features)

    # Render the result to HTML
    return render_template('index.html', prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True)
