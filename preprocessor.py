from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd

def preprocess_data(data):
    # Replace invalid entries
    data['occupation'] = data['occupation'].replace({'?': 'Others'})
    data['workclass'] = data['workclass'].replace({'?': 'NotListed'})
    
    # Drop rows with irrelevant workclass
    data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
    
    # Drop low-level education
    data = data[~data['education'].isin(['Preschool', '1st-4th', '5th-6th'])]
    
    # Drop redundant columns (this is why 'education' is not in the model!)
    data.drop(columns=['education'], inplace=True)
    
    # Remove outliers from age
    data = data[(data['age'] >= 17) & (data['age'] <= 75)]
    
    # Store encoders for each categorical column
    encoders = {}
    
    # Encode categorical columns and save encoders
    for column in data.select_dtypes(include='object').columns:
        if column != 'income':  # Don't encode the target variable here
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
            encoders[column] = encoder
    
    # Save the encoders for use in prediction
    joblib.dump(encoders, 'encoders.pkl')
    
    # Save the final column list (features used for training)
    feature_columns = [col for col in data.columns if col != 'income']
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    return data
