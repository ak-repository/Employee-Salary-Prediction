from data_loader import load_data
from explorer import explore_data
from preprocessor import preprocess_data
from model_trainer import train_model

# Step 1: Load data
data = load_data()

# Step 2: Explore data (optional)
explore_data(data)

# Step 3: Preprocess data
processed_data = preprocess_data(data)

# Step 4: Train model
train_model(processed_data)
