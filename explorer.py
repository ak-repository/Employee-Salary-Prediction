def explore_data(data):
    print("Head:\n", data.head(6))
    print("Tail:\n", data.tail(4))
    print("Shape:", data.shape)
    print("Missing Values:\n", data.isna().sum())
    
    print("\nValue Counts:")
    print("Occupation:\n", data.occupation.value_counts())
    print("Gender:\n", data.gender.value_counts())
    print("Marital Status:\n", data['marital-status'].value_counts())
    print("Education:\n", data['education'].value_counts())
    print("Workclass:\n", data['workclass'].value_counts())
