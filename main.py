from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset
data = pd.read_csv('Cleaned _data.csv')

# Load model
with open('RidgeModel .pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    location = request.form.get('location')
    bhk = request.form.get('BHK')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    # Check for empty inputs
    if not all([location, bhk, bath, sqft]):
        return "Error: One or more fields are empty."

    # Convert numeric fields safely
    try:
        bhk = float(bhk)
        bath = float(bath)
        sqft = float(sqft)
    except ValueError:
        return "Error: Please enter valid numbers for BHK, bath, and sqft."

    # Create DataFrame
    input_df = pd.DataFrame([[location, bhk, bath, sqft]],
                            columns=['location', 'BHK', 'bath', 'total_sqft'])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Return the result
    return f"Predicted House Price: {round(prediction, 2)}"

if __name__ == "__main__":
    app.run(debug=True, port=5004)
