import joblib

def predict(input_df):
    model = joblib.load("model.pkl")
    return model.predict(input_df)
