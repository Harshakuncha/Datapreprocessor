from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)

def preprocess_data(data):
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    imputer_num = SimpleImputer(strategy='mean')
    data[numerical_columns] = imputer_num.fit_transform(data[numerical_columns])
    imputer_cat = SimpleImputer(strategy='most_frequent')
    data[categorical_columns] = imputer_cat.fit_transform(data[categorical_columns])
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]), 
                                columns=encoder.get_feature_names_out(categorical_columns))
    
    # Scale numerical features
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data[numerical_columns]), 
                               columns=numerical_columns)
    processed_data = pd.concat([scaled_data, encoded_data], axis=1)
    
    return processed_data

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
    processed_data = preprocess_data(data)
    processed_data_html = processed_data.to_html(classes='table table-striped', border=0)
    return render_template('result.html', tables=[processed_data_html], titles=processed_data.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
