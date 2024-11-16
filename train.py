import pickle
import os
from sklearn.datasets import load_iris
from model import IrisModel

def train_and_save_model():
    """Train the IrisModel and save it as a pickle file"""
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create and train model
    model = IrisModel()
    model.train(X, y)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save trained model to pickle file
    with open('data/saved_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_model() 