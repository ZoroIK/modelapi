import os
import pickle
import requests
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel

# Use the public IBM Cloud Object Storage URL (replace with your actual model URL)
MODEL_URL = "https://rfc.s3.us-east.cloud-object-storage.appdomain.cloud/model.p"
MODEL_PATH = "model.p"

# Function to download the model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from IBM Cloud Object Storage...")
        response = requests.get(MODEL_URL, stream=True)

        # Check if the response is an error page
        if "text/html" in response.headers.get("Content-Type", ""):
            print("❌ Error: Downloaded an HTML page instead of the model. Check the model URL!")
            return False
        
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        print("✅ Model downloaded successfully!")

        # Verify file size
        if os.path.getsize(MODEL_PATH) < 100000:
            print("❌ Error: Model file is too small! Download might have failed.")
            return False

        return True

    return True

# Try downloading the model
if not download_model():
    exit(1)  # Exit if model download fails

# Load the model from the downloaded file
with open(MODEL_PATH, "rb") as f:
    try:
        model_dict = pickle.load(f)
        model = model_dict["model"]
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

# FastAPI App
app = FastAPI()

# Pydantic model for incoming request
class LandmarkData(BaseModel):
    landmarks: list

# Prediction endpoint
@app.post("/predict")
async def predict(data: LandmarkData):
    if len(data.landmarks) < 42:
        raise HTTPException(status_code=400, detail="Insufficient landmark data")
    
    try:
        # Assuming the model needs only the first 42 landmarks
        prediction = model.predict([np.array(data.landmarks)[:42]])[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point to run the FastAPI app locally
if __name__ == "__main__":
    import uvicorn
    # Run locally on localhost with port 8080
    uvicorn.run(app, host="127.0.0.1", port=5000)
