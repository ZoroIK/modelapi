from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
import os
import requests
import joblib
# IBM Cloud Object Storage URL (Replace with your actual URL)
MODEL_URL = "https://s3.us-east.cloud-object-storage.appdomain.cloud/rfc/model.p"
MODEL_PATH = "model_compressed.p"

# Function to download the model if it's not already present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from IBM Cloud...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Model downloaded successfully!")

# Download and load the model
download_model()
with open(MODEL_PATH, "rb") as f:
    model_dict = joblib.load(f)
    model = model_dict["model"]

# Define API app
app = FastAPI()

# Define request body model
class LandmarkData(BaseModel):
    landmarks: list

@app.post("/predict")
async def predict(data: LandmarkData):
    if len(data.landmarks) < 42:
        raise HTTPException(status_code=400, detail="Insufficient landmark data")
    
    try:
        prediction = model.predict([np.array(data.landmarks)[:42]])[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server (only for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)