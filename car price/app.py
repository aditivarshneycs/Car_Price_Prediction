# app.py

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load the cleaned car data to get unique values for dropdowns
car = pd.read_csv(r'C:\Users\HP\Downloads\car price\car price\Cleaned_Car_data.csv')

@app.route('/', methods=['GET'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())
    
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    # Debugging: Print received form data
    print("Received data:", data)
    
    company = data.get('company')
    car_model = data.get('car_models')  # Ensure this matches the form field name
    year = data.get('year')
    fuel_type = data.get('fuel_type')
    kilo_driven = data.get('kilo_driven')
    
    # Check if any field is missing
    if not all([company, car_model, year, fuel_type, kilo_driven]):
        return jsonify({'error': 'Missing data. Please fill all fields.'}), 400
    
    try:
        year = int(year)
    except ValueError:
        return jsonify({'error': 'Invalid year format.'}), 400
    
    try:
        kilo_driven = int(kilo_driven)
    except ValueError:
        return jsonify({'error': 'Invalid kilometers driven format.'}), 400

    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'name': [car_model],
        'company': [company],
        'year': [year],
        'kms_driven': [kilo_driven],
        'fuel_type': [fuel_type]
    })
    
    # Predict using the loaded model
    try:
        prediction = model.predict(input_data)[0]
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'Error in prediction.'}), 500
    
    # Format prediction
    predicted_price = f"₹{np.round(prediction, 2):,.2f}"
    
    return jsonify({'prediction': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
