import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The GDP of the Country for Population {} Area_sqm {} Pop_Density per Sqm {} Net Migration {} and Literacy {} should be $ {}'.format(final_features[0],final_features[1],final_features[2],final_features[3],final_features[4],output))

if __name__ == "__main__":
    app.run(debug=True)
