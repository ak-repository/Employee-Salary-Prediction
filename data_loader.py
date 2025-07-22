import pandas as pd

def load_data():
    data = pd.read_csv("adult_3.csv")  # make sure the file is in the same folder
    return data
