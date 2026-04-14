from flask import Flask, request, jsonify
import joblib
import numpy as np

# 1. Load saved model
model = joblib.load("iris_model.pkl")

# 2. Create Flask app
app = Flask(__name__)

# 3. Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  # expects a list of features
    prediction = model.predict([np.array(data)])
    return jsonify({'prediction': int(prediction[0])})

# 4. Run server
if __name__ == '__main__':
    app.run(debug=True)
