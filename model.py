from sklearn.ensemble import RandomForestClassifier
import numpy as np

class IrisModel:
    def __init__(self):
        """Initialize the IrisModel with a RandomForestClassifier"""
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42
        )
        self.feature_names = [
            'sepal_length', 
            'sepal_width', 
            'petal_length', 
            'petal_width'
        ]
        self.target_names = ['setosa', 'versicolor', 'virginica']
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model with given features and targets
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
        """
        self.model.fit(X, y)
    
    def predict(self, features: list) -> dict:
        """
        Predict iris class from input features
        
        Args:
            features (list): List of 4 float values representing iris measurements
            
        Returns:
            dict: Prediction results including class and probability
        """
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0]
        return {
            'predicted_class': self.target_names[prediction],
            'probability': float(max(probability))
        } 