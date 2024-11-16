from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import os

app = FastAPI(
    title="Iris Classifier API",
    description="API for classifying iris flowers using machine learning"
)

# Load the pickled model
MODEL_PATH = 'data/saved_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception(f"Model file not found at {MODEL_PATH}. Please run train.py first.")

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

@app.post("/predict", response_model=dict)
async def predict(features: IrisFeatures):
    """
    Predict iris species from flower measurements
    
    Args:
        features (IrisFeatures): Iris flower measurements
        
    Returns:
        dict: Prediction results including species and confidence
    """
    try:
        feature_list = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]
        
        result = model.predict(feature_list)
        return {
            "status": "success",
            "prediction": result["predicted_class"],
            "probability": result["probability"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 