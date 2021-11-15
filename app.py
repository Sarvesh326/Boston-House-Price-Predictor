from flask import Flask, render_template, request
from joblib import load
import numpy as np

model = load(open('Dragon.joblib', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
    d1 = request.form['1']
    d2 = request.form['2']
    d3 = request.form['3']
    d4 = request.form['4']
    d5 = request.form['5']
    d6 = request.form['6']
    d7 = request.form['7']
    d8 = request.form['8']
    d9 = request.form['9']
    d10 = request.form['10']
    d11 = request.form['11']
    d12 = request.form['12']
    d13 = request.form['13']
    arr = np.array([[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13]])
    pred = model.predict(arr)
    output = round(pred[0],2)
    return render_template('index.html', prediction_text="Estimated Price in $1000: {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
