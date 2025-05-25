
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import pdfplumber
import shutil
from pathlib import Path
import pickle
from pymongo import MongoClient
import uvicorn
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from fastapi import Query
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

with open("college_model.pkl", "rb") as f:
    model, encoder, label_encoder, df = joblib.load(f)


@app.post("/extract")
async def extract_tables_from_pdf(file: UploadFile = File(...)):
    """Extract tables from uploaded PDF file"""
    pdf_path = UPLOAD_DIR / file.filename

    # Save uploaded file
    with pdf_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    all_data = []
    first_table_header = None  # Store header of the first detected table

    # Open and process PDF
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages: 
            tables = page.extract_tables()
            for table in tables:
                if table:  # Ensure table is not empty
                    df = pd.DataFrame(table)

                    # If it's the first detected table, save its header
                    if first_table_header is None:
                        first_table_header = df.iloc[0]  # Assume first row is header
                        df = df[1:]  # Remove header from data

                    df.columns = first_table_header  # Apply consistent headers
                    df.reset_index(drop=True, inplace=True)
                    all_data.append(df)

    # Convert extracted tables to JSON
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        data_json = final_df.to_dict(orient='records')
        return {"data": data_json}
    else:
        return {"message": "No tables found in the PDF."}

class PredictRequest(BaseModel):
    course: str
    category: str
    rank: int
    ctype: str
@app.post("/predict")

async def predict(request: PredictRequest):
    """Predict multiple recommended colleges using XGBoost and return ranks"""
    try:
        # One-Hot Encode Inputs
        input_data = pd.DataFrame([[request.ctype, request.category, request.course]], 
                                  columns=["Type", "Category", "courseName"])
        input_encoded = encoder.transform(input_data)

        # Add Rank Feature
        input_final = np.hstack((input_encoded, [[request.rank]]))

        # Predict probabilities for all colleges
        probabilities = model.predict_proba(input_final)[0]

        # Get top N predictions
        top_n = 5
        top_indices = np.argsort(probabilities)[-top_n:][::-1]

        # Convert indices back to college names
        predicted_colleges = label_encoder.inverse_transform(top_indices)

        # Fetch actual ranks for the predicted colleges
        college_ranks = df[
            (df["cname"].isin(predicted_colleges)) & 
            (df["Category"] == request.category) & 
            (df["courseName"] == request.course)
        ][["cname", "Rank"]].drop_duplicates()

        #  Convert to list of dictionaries
        recommended_colleges = [
            {"college": row["cname"], "rank": int(row["Rank"])}
            for _, row in college_ranks.iterrows()
        ]

        return {"recommended_colleges": recommended_colleges}

    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
 
  