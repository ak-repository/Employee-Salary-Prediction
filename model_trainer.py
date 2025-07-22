from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model(data):
    # Encode the target variable
    target_encoder = LabelEncoder()
    data['income'] = target_encoder.fit_transform(data['income'])
    
    # Save target encoder
    joblib.dump(target_encoder, 'target_encoder.pkl')
    
    X = data.drop("income", axis=1)
    y = data["income"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    joblib.dump(model, "model.pkl")  # Save the model
    
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Print the feature columns the model was trained on
    print("Features used for training:", list(X.columns))
