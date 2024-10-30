from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import io
    import base64
except ImportError:
    plt = None  # Handle case where matplotlib is not installed

app = Flask(__name__)

# Define the path to your model
model_path = 'model/stroke3_model.pkl'

# Load the pre-trained Random Forest model using pickle
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('form.html', prediction_result='')

# Route for single input prediction
@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        # Gender mapping
        gender_male = 0
        gender_female = 0

        gender_status = request.form['gender']
        if gender_status.lower() == 'male':
            gender_male = 1
        elif gender_status.lower() == 'female':
            gender_female = 1

        # Smoking status mapping
        smoking_status = request.form['smoking_status']
        smoking_status_Unknown = 0
        smoking_status_formerly_smoked = 0
        smoking_status_never_smoked = 0
        smoking_status_smokes = 0

        if smoking_status == 'unknown':
            smoking_status_Unknown = 1
        elif smoking_status == 'formerly_smoked':
            smoking_status_formerly_smoked = 1
        elif smoking_status == 'never_smoked':
            smoking_status_never_smoked = 1
        elif smoking_status == 'smokes':
            smoking_status_smokes = 1

        # Collect the rest of the form data
        inputs = [
            float(request.form['age']),
            float(request.form['hypertension']),
            float(request.form['heart_disease']),
            1 if request.form['ever_married'].lower() == 'yes' else 0,
            1 if request.form['Residence_type'].lower() == 'urban' else 0,
            float(request.form['avg_glucose_level']),
            float(request.form['bmi']),
            gender_male,
            gender_female,
            float(request.form['work_type_Govt_job']),
            float(request.form['work_type_Private']),
            float(request.form['work_type_Self-employed']),
            float(request.form['work_type_children']),
            smoking_status_Unknown,
            smoking_status_formerly_smoked,
            smoking_status_never_smoked,
            smoking_status_smokes
        ]
    except ValueError:
        return "Please enter valid values for all fields."

    # Reshape the input to match model input format
    input_array = np.array(inputs).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_array)[0]

    # Display result in HTML
    result = f"<h3>Single Prediction Result: {'Stroke' if prediction == 1 else 'No Stroke'}</h3>"

    # Render the form and display the prediction result
    return render_template('form.html', prediction_result=result)

# Route for batch prediction via CSV
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Read the CSV file
        data = pd.read_csv(file)

        # Check for the required columns
        expected_columns = ['age', 'hypertension', 'heart_disease', 'ever_married',
                            'Residence_type', 'avg_glucose_level', 'bmi', 'Male', 'Female',
                            'work_type_Govt_job', 'work_type_Private', 'work_type_Self-employed',
                            'work_type_children', 'smoking_status_Unknown',
                            'smoking_status_formerly smoked', 'smoking_status_never smoked',
                            'smoking_status_smokes']

        if not all(column in data.columns for column in expected_columns):
            return "CSV does not contain the required columns."

        # Prepare data for prediction
        input_data = data[expected_columns]
        input_data['ever_married'] = input_data['ever_married'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
        input_data['Residence_type'] = input_data['Residence_type'].apply(lambda x: 1 if str(x).lower() == 'urban' else 0)

        # Make batch predictions
        predictions = model.predict(input_data)
        data['Prediction'] = ['Stroke' if pred == 1 else 'No Stroke' for pred in predictions]

        # Generate bar graph if matplotlib is available
        plot_url = None
        if plt:
            # Count the number of strokes (1) and no strokes (0)
            stroke_counts = data['Prediction'].value_counts()

            # Generate the bar graph
            plt.figure(figsize=(8, 6))
            stroke_counts.plot(kind='bar', color=['skyblue', 'salmon'])
            plt.title('Count of Stroke Predictions')
            plt.xlabel('Prediction (No Stroke, Stroke)')
            plt.ylabel('Count')
            plt.xticks(rotation=0)

            # Save the plot to a bytes object and encode it in base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

        # Convert the result to an HTML table
        result_table = data.to_html(classes='data', header="true", index=False)

        # Render the form and display the batch prediction result with the bar graph (if available)
        return render_template('form.html', prediction_result=result_table, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
