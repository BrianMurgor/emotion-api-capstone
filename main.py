from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and encoder
model = joblib.load("emotion_model.pkl")
encoder = joblib.load("label_encoder.pkl")

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_emotion(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    emotion = encoder.inverse_transform([prediction])[0]
    return {"emotion": emotion}

