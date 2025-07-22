import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load all necessary components
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('model.pkl')
        encoders = joblib.load('encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        return model, encoders, feature_columns, target_encoder
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please run the training pipeline first.")
        return None, None, None, None

def preprocess_input(age, workclass, occupation, marital_status, hours_per_week, encoders, feature_columns):
    """
    Preprocess input data to match the training format
    Note: 'education' is NOT included because it was dropped during training
    """
    # Create DataFrame with the features that were actually used in training
    # Based on your preprocessor.py, these are the likely remaining columns after dropping 'education'
    input_data = {
        'age': age,
        'workclass': workclass,
        'occupation': occupation,
        'marital-status': marital_status,
        'hours-per-week': hours_per_week
    }
    
    df = pd.DataFrame([input_data])
    
    # Apply the same preprocessing steps
    # Replace invalid entries (same as in preprocessor.py)
    df['occupation'] = df['occupation'].replace({'?': 'Others'})
    df['workclass'] = df['workclass'].replace({'?': 'NotListed'})
    
    # Encode categorical columns using the saved encoders
    for column in df.select_dtypes(include='object').columns:
        if column in encoders:
            try:
                df[column] = encoders[column].transform(df[column])
            except ValueError:
                # Handle unseen categories
                st.warning(f"Unknown category '{df[column].iloc[0]}' in {column}. Using most frequent category.")
                # Use the first class from the encoder as fallback
                df[column] = 0
    
    # Ensure columns match training features
    df = df[feature_columns]
    
    return df

# Page configuration
st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="🧑‍💼",
    layout="wide"
)

# Title and description
st.title("🧑‍💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Load model components
model, encoders, feature_columns, target_encoder = load_model_components()

if model is not None:
    # Display which features are actually used
    st.info(f"**Features used by the model:** {', '.join(feature_columns)}")
    
    # Create sidebar inputs (remove 'education' since it's not used!)
    with st.sidebar:
        st.header("Input Employee Details")
        
        age = st.slider("Age", min_value=17, max_value=75, value=22, step=1)
        
        # Workclass options (based on your preprocessor)
        workclass_options = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                           "Local-gov", "State-gov", "NotListed"]
        workclass = st.selectbox("Work Class", workclass_options, index=0)
        
        # Occupation options
        occupation_options = ["Tech-support", "Craft-repair", "Other-service", "Sales",
                            "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                            "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                            "Transport-moving", "Priv-house-serv", "Protective-serv", 
                            "Armed-Forces", "Others"]
        occupation = st.selectbox("Occupation", occupation_options, index=0)
        
        # Marital status
        marital_options = ["Never-married", "Married-civ-spouse", "Divorced", "Married-spouse-absent",
                         "Separated", "Married-AF-spouse", "Widowed"]
        marital_status = st.selectbox("Marital Status", marital_options, index=0)
        
        hours_per_week = st.slider("Hours per week", min_value=1, max_value=80, value=40, step=1)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Input Data")
        
        # Show the actual features being used
        display_data = {
            "age": [age],
            "workclass": [workclass],
            "occupation": [occupation],
            "marital-status": [marital_status],
            "hours-per-week": [hours_per_week]
        }
        
        display_df = pd.DataFrame(display_data)
        st.table(display_df)
        
        # Predict button
        if st.button("Predict Salary Class", type="primary"):
            try:
                # Preprocess input
                processed_input = preprocess_input(age, workclass, occupation, marital_status, 
                                                 hours_per_week, encoders, feature_columns)
                
                # Make prediction
                prediction = model.predict(processed_input)
                prediction_proba = model.predict_proba(processed_input)
                
                # Convert prediction back to original labels
                prediction_label = target_encoder.inverse_transform(prediction)[0]
                
                # Display result
                if prediction[0] == 1:
                    st.success(f"🎉 Prediction: {prediction_label}")
                    st.info(f"Confidence: {prediction_proba[0][1]:.2%}")
                else:
                    st.info(f"📊 Prediction: {prediction_label}")
                    st.info(f"Confidence: {prediction_proba[0][0]:.2%}")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Debug info:", str(e))

    with col2:
        st.subheader("Model Information")
        st.write("**Model Type:** Logistic Regression")
        st.write(f"**Features:** {', '.join(feature_columns)}")
        st.write("**Target:** Salary >50K or ≤50K")
        st.write("**Note:** Education column was dropped during training")

    # Updated Batch Prediction Section
    st.markdown("---")
    st.subheader("Batch Prediction")
    
    # Show expected format
    with st.expander("📋 Expected CSV Format"):
        st.write("Your CSV file should contain these columns (matching training data):")
        expected_df = pd.DataFrame({
            'age': [22, 35],
            'workclass': ['Private', 'Self-emp-inc'], 
            'occupation': ['Tech-support', 'Sales'],
            'marital-status': ['Never-married', 'Married-civ-spouse'],
            'hours-per-week': [40, 45]
        })
        st.dataframe(expected_df)
        st.warning("⚠️ Do NOT include 'education' or 'income' columns - they are not used by the model!")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            
            st.write("**Original uploaded data:**")
            st.dataframe(batch_data.head())
            
            # Remove columns that shouldn't be there
            unwanted_columns = ['education', 'income', 'salary', 'target']
            for col in unwanted_columns:
                if col in batch_data.columns:
                    batch_data = batch_data.drop(columns=[col])
                    st.info(f"Removed column '{col}' as it's not used by the model")
            
            # Check if we have the required columns
            missing_cols = set(feature_columns) - set(batch_data.columns)
            if missing_cols:
                st.error(f"Missing required columns: {list(missing_cols)}")
            else:
                if st.button("Run Batch Prediction"):
                    try:
                        # Apply the same preprocessing to batch data
                        processed_batch = batch_data.copy()
                        
                        # Apply preprocessing steps
                        processed_batch['occupation'] = processed_batch['occupation'].replace({'?': 'Others'})
                        processed_batch['workclass'] = processed_batch['workclass'].replace({'?': 'NotListed'})
                        
                        # Encode categorical columns
                        for column in processed_batch.select_dtypes(include='object').columns:
                            if column in encoders:
                                try:
                                    processed_batch[column] = encoders[column].transform(processed_batch[column])
                                except ValueError as e:
                                    st.warning(f"Some unknown categories in {column}, using fallback values")
                                    # Handle unknown categories by mapping to 0
                                    unknown_mask = ~processed_batch[column].isin(encoders[column].classes_)
                                    processed_batch.loc[unknown_mask, column] = encoders[column].classes_[0]
                                    processed_batch[column] = encoders[column].transform(processed_batch[column])
                        
                        # Ensure column order matches training
                        processed_batch = processed_batch[feature_columns]
                        
                        # Make predictions
                        predictions = model.predict(processed_batch)
                        prediction_probas = model.predict_proba(processed_batch)
                        
                        # Add results to original dataframe
                        results_df = batch_data.copy()
                        results_df['Prediction'] = target_encoder.inverse_transform(predictions)
                        results_df['Confidence'] = [f"{max(proba):.2%}" for proba in prediction_probas]
                        
                        st.success(f"✅ Predictions completed for {len(results_df)} records!")
                        st.dataframe(results_df)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download predictions as CSV",
                            data=csv,
                            file_name="salary_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

else:
    st.error("Model components not loaded. Please run the training pipeline first.")
