# Emotion Detection API – Capstone Project

This project provides an API to predict emotional states based on audio features using a machine learning model.

## Setup Instructions (Local)

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
uvicorn main:app --reload
```

4. Open in browser:
```
http://127.0.0.1:8000/docs
```

##  API Endpoint

### POST /predict

**Input:**
```json
{
  "features": [12.4, 13.5, 10.1, 9.7, 8.5, 7.2, 6.8, 11.0, 10.5, 9.1, 8.3, 7.7, 6.9]
}
```

**Output:**
```json
{
  "emotion": "amuzed"
}
```

## Files Explained

- `main.py`: FastAPI app with prediction endpoint
- `emotion_model.pkl`: Trained scikit-learn model
- `label_encoder.pkl`: Label encoder for emotion labels
- `requirements.txt`: Python dependencies
- `Dockerfile`: (Optional) for cloud deployment
