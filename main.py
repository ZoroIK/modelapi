import os
import pickle
import requests
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel

# Use the public Google Cloud Storage URL
MODEL_URL = "https://storage.googleapis.com/rcfmodel/model.p"
MODEL_PATH = "model.p"

# Function to download the model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Cloud Storage...")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
