from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Function to preprocess data
def preprocess_data(data):
    # Handle missing values
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    # Impute missing numerical values with mean
    imputer_num = SimpleImputer(strategy='mean')
    data[numerical_columns] = imputer_num.fit_transform(data[numerical_columns])
    
    # Impute missing categorical values with the most frequent value
    imputer_cat = SimpleImputer(strategy='most_frequent')
    data[categorical_columns] = imputer_cat.fit_transform(data[categorical_columns])
    
    # One-Hot Encoding for categorical columns
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]), 
                                columns=encoder.get_feature_names_out(categorical_columns))
    
    # Scale numerical features
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data[numerical_columns]), 
                               columns=numerical_columns)
    
    # Combine scaled numerical data and encoded categorical data
    processed_data = pd.concat([scaled_data, encoded_data], axis=1)
    
    return processed_data

# Route to display the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and data preprocessing
@app.route('/preprocess', methods=['POST'])
def preprocess():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Read the uploaded CSV file into a DataFrame
    data = pd.read_csv(file)
    
    # Perform data preprocessing
    processed_data = preprocess_data(data)
    
    # Convert processed data to HTML table format
    processed_data_html = processed_data.to_html(classes='table table-striped', border=0)
    
    # Render the results page with the processed data
    return render_template('result.html', tables=[processed_data_html], titles=processed_data.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
