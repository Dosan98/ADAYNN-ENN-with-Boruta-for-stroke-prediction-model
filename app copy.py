from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Define the path to your model
model_path = 'model/random_forest_model.pkl'

# Load the pre-trained Random Forest model using pickle
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('form.html')  # Render a form to accept 16 inputs

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all form inputs
        inputs = [
            float(request.form['gender']),
            float(request.form['age']),
            float(request.form['hypertension']),
            float(request.form['heart_disease']),
            float(request.form['ever_married']),
            float(request.form['Residence_type']),
            float(request.form['avg_glucose_level']),
            float(request.form['bmi']),
            float(request.form['work_type_Govt_job']),
            float(request.form['work_type_Private']),
            float(request.form['work_type_Self-employed']),
            float(request.form['work_type_children']),
            float(request.form['smoking_status_Unknown']),
            float(request.form['smoking_status_formerly_smoked']),
            float(request.form['smoking_status_never_smoked']),
            float(request.form['smoking_status_smokes'])
        ]
    except ValueError:
        return "Please enter valid numbers for all 16 variables."

    # Convert list to numpy array and reshape for prediction (assuming model expects 2D input)
    input_array = np.array(inputs).reshape(1, -1)

    # Make a prediction using the Random Forest model
    prediction = model.predict(input_array)

    # Return the prediction result
    return f"The predicted value is: {prediction[0]}"

if __name__ == '__main__':
    app.run(debug=True)

