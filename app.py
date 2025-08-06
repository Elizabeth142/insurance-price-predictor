from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

model = None
poly_features = None
feature_names = ['age', 'bmi', 'children', 'smoker']

def create_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    age = np.random.randint(18, 65, n_samples)
    bmi = np.random.normal(28, 6, n_samples)
    bmi = np.clip(bmi, 15, 50) 
    children = np.random.poisson(1, n_samples)
    children = np.clip(children, 0, 5)
    smoker = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    charges = (
        age * 250 +
        (bmi - 25) * 100 +
        children * 500 +
        smoker * 20000 +
        np.random.normal(0, 3000, n_samples)
    )
    charges = np.clip(charges, 1000, 60000) 
    
    data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'charges': charges
    }
    
    return pd.DataFrame(data)

def train_model():
   
    global model, poly_features
    

    df = create_sample_data()
    
    X = df[['age', 'bmi', 'children', 'smoker']]
    y = df['charges']
    
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    
   
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)
  
    model = LinearRegression()
    model.fit(X_train, y_train)

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
    
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/model-info')
def model_info():
    
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
 
    try:
        if model is None:
            return jsonify({"error": "Model not trained"}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_fields = ['age', 'bmi', 'children', 'smoker']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}"
            }), 400
    
        try:
            age = float(data['age'])
            bmi = float(data['bmi'])
            children = int(data['children'])
            smoker = int(data['smoker'])
        except (ValueError, TypeError):
            return jsonify({
                "error": "Invalid data types. Age and BMI should be numbers, children and smoker should be integers"
            }), 400
        
      
        if not (18 <= age <= 64):
            return jsonify({"error": "Age must be between 18 and 64"}), 400
        
        if not (15 <= bmi <= 50):
            return jsonify({"error": "BMI must be between 15 and 50"}), 400
        
        if not (0 <= children <= 5):
            return jsonify({"error": "Children must be between 0 and 5"}), 400
        
        if smoker not in [0, 1]:
            return jsonify({"error": "Smoker must be 0 (non-smoker) or 1 (smoker)"}), 400
      
        input_data = np.array([[age, bmi, children, smoker]])
        input_poly = poly_features.transform(input_data)
        
     
        prediction = model.predict(input_poly)[0]
      
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
               
                age = float(record['age'])
                bmi = float(record['bmi'])
                children = int(record['children'])
                smoker = int(record['smoker'])
                
              
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
  
    print("Training model...")
    train_model()
    print("Starting Flask API...")

    app.run(debug=True, host='0.0.0.0', port=5000)