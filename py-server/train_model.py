# import pandas as pd
# import joblib
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from pymongo import MongoClient

# def fetch_data_from_mongodb():
#     """Fetch admission data from MongoDB"""
#     client = MongoClient("mongodb+srv://admin:admin@cluster0.eqevsdm.mongodb.net/keam?retryWrites=true&w=majority&appName=Cluster0")
#     db = client["keam"]
#     collection = db["colleges"]
#     data = list(collection.find())
#     return pd.DataFrame(data)

# def preprocess_data(df):
#     """Preprocesses data: encoding categorical columns, handling NaNs, and structuring rank"""
#     df = df.drop(columns=["_id", "ccode", "__v"], errors='ignore')

#     # Convert to long format (if category columns are separate)
#     df = df.melt(id_vars=["Type", "cname", "courseName"], var_name="Category", value_name="Rank")

#     df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
#     df.dropna(subset=["Rank"], inplace=True)  

#     return df

# def train_model(df):
#     """Encodes categorical variables, scales data, trains KNN, and saves the model"""
#     le_type = LabelEncoder()
#     le_category = LabelEncoder()
#     le_course = LabelEncoder()

#     df["Type"] = le_type.fit_transform(df["Type"])
#     df["Category"] = le_category.fit_transform(df["Category"])
#     df["courseName"] = le_course.fit_transform(df["courseName"])

#     # Scale the features
#     scaler = StandardScaler()
#     X = df[["Type", "Category", "courseName", "Rank"]]
#     X_scaled = scaler.fit_transform(X)

#     # Train Nearest Neighbors Model
#     model = NearestNeighbors(n_neighbors=10, metric="euclidean")
#     model.fit(X_scaled)

#     # Save the trained model, encoders, scaler, and college mapping
#     joblib.dump((model, le_type, le_category, le_course, scaler, df[["cname", "Rank"]]), "college_model.pkl")

#     print("Model training complete! Saved as 'college_model.pkl'.")

# if __name__ == "__main__":
#     print("Fetching data from MongoDB...")
#     df = fetch_data_from_mongodb()

#     print(" Preprocessing data...")
#     df = preprocess_data(df)
#     print(f" Data preprocessed: {df.shape[0]} rows available for training.")

#     print(" Training model...")
#     train_model(df)

# import pandas as pd
# import joblib
# import numpy as np
# from pymongo import MongoClient
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# import xgboost as xgb

# # Fetch Data from MongoDB
# def fetch_data_from_mongodb():
#     client = MongoClient("mongodb+srv://admin:admin@cluster0.eqevsdm.mongodb.net/keam?retryWrites=true&w=majority&appName=Cluster0")
#     db = client["keam"]
#     collection = db["colleges"]
#     data = list(collection.find())
#     return pd.DataFrame(data)

# #  Preprocess Data
# def preprocess_data(df):
#     df = df.drop(columns=["_id", "ccode", "__v"], errors='ignore')
#     df = df.melt(id_vars=["Type", "cname", "courseName"], var_name="Category", value_name="Rank")
#     df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
#     df.dropna(subset=["Rank"], inplace=True)

#     # Remove classes with only one occurrence
#     college_counts = df["cname"].value_counts()
#     df = df[df["cname"].isin(college_counts[college_counts > 1].index)]
    
#     return df

# #  Train Model
# def train_model(df):
#     # One-hot encode categorical features
#     encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#     categorical_features = df[["Type", "Category", "courseName"]]
#     X_categorical = encoder.fit_transform(categorical_features)
    
#     # Extract numerical features
#     X_numerical = df[["Rank"]].values
#     X = np.hstack((X_categorical, X_numerical))

#     # Encode college names before splitting
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["cname"])  

#     # Split data into training and testing sets (without stratify)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train XGBoost model
#     model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
#     model.fit(X_train, y_train)

#     # Save model, encoder, and label encoder
#     joblib.dump((model, encoder, label_encoder), "college_model.pkl")
#     print(" Model training complete! Saved as 'college_model.pkl'.")

# # Execute Training Process
# if __name__ == "__main__":
#     print(" Fetching data from MongoDB...")
#     df = fetch_data_from_mongodb()
#     print(" Preprocessing data...")
#     df = preprocess_data(df)
#     print(f" Data preprocessed: {df.shape[0]} rows available for training.")
#     print("Training model...")
#     train_model(df)


import pandas as pd
import joblib
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb

# Fetch Data from MongoDB
def fetch_data_from_mongodb():
    client = MongoClient("mongodb+srv://admin:admin@cluster0.eqevsdm.mongodb.net/keam?retryWrites=true&w=majority&appName=Cluster0")
    db = client["keam"]
    collection = db["colleges"] 
    data = list(collection.find())
    return pd.DataFrame(data)

#  Preprocess Data
def preprocess_data(df):
    df = df.drop(columns=["_id", "ccode", "__v"], errors='ignore')
    df = df.melt(id_vars=["Type", "cname", "courseName"], var_name="Category", value_name="Rank")
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df.dropna(subset=["Rank"], inplace=True)

    # Remove classes with only one occurrence
    college_counts = df["cname"].value_counts()
    df = df[df["cname"].isin(college_counts[college_counts > 1].index)]
    
    return df

#  Train Model
def train_model(df):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_features = df[["Type", "Category", "courseName"]]
    X_categorical = encoder.fit_transform(categorical_features)
    
    X_numerical = df[["Rank"]].values
    X = np.hstack((X_categorical, X_numerical))

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["cname"])  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    #  Save model, encoder, label encoder, and original df
    joblib.dump((model, encoder, label_encoder, df), "college_model.pkl")
    print(" Model training complete! Saved as 'college_model.pkl'.")

# 🔄 Execute Training Process
if __name__ == "__main__":
    print(" Fetching data from MongoDB...")
    df = fetch_data_from_mongodb()
    print(" Preprocessing data...")
    df = preprocess_data(df)
    print(f" Data preprocessed: {df.shape[0]} rows available for training.")
    print("🚀 Training model...")
    train_model(df)
