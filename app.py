from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store the trained model and preprocessors
model = None
poly_features = None
feature_names = ['age', 'bmi', 'children', 'smoker']

def create_sample_data():
    """Create sample insurance data for training the model"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data similar to insurance dataset
    age = np.random.randint(18, 65, n_samples)
    bmi = np.random.normal(28, 6, n_samples)
    bmi = np.clip(bmi, 15, 50)  # Realistic BMI range
    children = np.random.poisson(1, n_samples)
    children = np.clip(children, 0, 5)
    smoker = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Create charges based on realistic relationships
    charges = (
        age * 250 +
        (bmi - 25) * 100 +
        children * 500 +
        smoker * 20000 +
        np.random.normal(0, 3000, n_samples)
    )
    charges = np.clip(charges, 1000, 60000)  # Realistic charge range
    
    data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'charges': charges
    }
    
    return pd.DataFrame(data)

def train_model():
    """Train the polynomial regression model"""
    global model, poly_features
    
    # Create sample data
    df = create_sample_data()
    
    # Prepare features (excluding sex and region as in your final model)
    X = df[['age', 'bmi', 'children', 'smoker']]
    y = df['charges']
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Model trained successfully!")
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Testing R² Score: {test_score:.4f}")

from flask import render_template


@app.route('/')
def home_page():
    return render_template('index.html')

def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Insurance Charges Prediction API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict insurance charges",
            "/health": "GET - Check API health",
            "/model-info": "GET - Get model information"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/model-info')
def model_info():
    """Get information about the trained model"""
    if model is None:
        return jsonify({"error": "Model not trained"}), 500
    
    return jsonify({
        "model_type": "Polynomial Linear Regression",
        "polynomial_degree": 2,
        "features": feature_names,
        "feature_descriptions": {
            "age": "Age of the insured person (18-64)",
            "bmi": "Body Mass Index (15.0-50.0)",
            "children": "Number of children/dependents (0-5)",
            "smoker": "Smoking status (0: Non-smoker, 1: Smoker)"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict insurance charges based on input features"""
    try:
        if model is None:
            return jsonify({"error": "Model not trained"}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['age', 'bmi', 'children', 'smoker']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}"
            }), 400
        
        # Validate data types and ranges
        try:
            age = float(data['age'])
            bmi = float(data['bmi'])
            children = int(data['children'])
            smoker = int(data['smoker'])
        except (ValueError, TypeError):
            return jsonify({
                "error": "Invalid data types. Age and BMI should be numbers, children and smoker should be integers"
            }), 400
        
        # Validate ranges
        if not (18 <= age <= 64):
            return jsonify({"error": "Age must be between 18 and 64"}), 400
        
        if not (15 <= bmi <= 50):
            return jsonify({"error": "BMI must be between 15 and 50"}), 400
        
        if not (0 <= children <= 5):
            return jsonify({"error": "Children must be between 0 and 5"}), 400
        
        if smoker not in [0, 1]:
            return jsonify({"error": "Smoker must be 0 (non-smoker) or 1 (smoker)"}), 400
        
        # Prepare input for prediction
        input_data = np.array([[age, bmi, children, smoker]])
        input_poly = poly_features.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_poly)[0]
        
        # Ensure prediction is positive
        prediction = max(prediction, 0)
        
        return jsonify({
            "prediction": round(prediction, 2),
            "input": {
                "age": age,
                "bmi": bmi,
                "children": children,
                "smoker": "Yes" if smoker == 1 else "No"
            },
            "currency": "USD"
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
from flask_cors import CORS
CORS(app)    

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Predict insurance charges for multiple records"""
    try:
        if model is None:
            return jsonify({"error": "Model not trained"}), 500
        
        data = request.get_json()
        
        if not data or 'records' not in data:
            return jsonify({"error": "No records provided. Send data as {'records': [...]}"})
        
        records = data['records']
        predictions = []
        
        for i, record in enumerate(records):
            try:
                # Validate and extract features
                age = float(record['age'])
                bmi = float(record['bmi'])
                children = int(record['children'])
                smoker = int(record['smoker'])
                
                # Prepare input and predict
                input_data = np.array([[age, bmi, children, smoker]])
                input_poly = poly_features.transform(input_data)
                prediction = model.predict(input_poly)[0]
                prediction = max(prediction, 0)
                
                predictions.append({
                    "record_id": i,
                    "prediction": round(prediction, 2),
                    "input": record
                })
                
            except Exception as e:
                predictions.append({
                    "record_id": i,
                    "error": f"Failed to process record: {str(e)}",
                    "input": record
                })
        
        return jsonify({
            "predictions": predictions,
            "total_records": len(records),
            "successful_predictions": len([p for p in predictions if 'prediction' in p])
        })
        
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Train the model on startup
    print("Training model...")
    train_model()
    print("Starting Flask API...")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)