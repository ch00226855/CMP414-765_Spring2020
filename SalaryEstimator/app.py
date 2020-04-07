import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

data = pd.DataFrame({
    'Experience': [1, 2, 3, 4, 5, 4, 3],
    'Test_Score': [95, 38, 56, 28, 69, 85, 100],
    'Interview_Score': [9, 7, 6, 4, 8, 6, 4],
    'Salary': [67000, 85000, 100000, 49000, 93000, 120000, 95000]
})

model = LinearRegression()
model.fit(data[['Experience', 'Test_Score', 'Interview_Score']], data['Salary'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = np.array([int(x) for x in request.form.values()]).reshape([1, -1])
    prediction = model.predict(features)
    return render_template('index.html', prediction_text='Employee salary should be $ {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)