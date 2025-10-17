# src/models/element_predictor.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
from typing import Dict, List, Tuple, Optional

class ElementPredictor:
    """Predict elements present in a sample from mass spectrometry data."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.element_labels = None
        self.feature_columns = None
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_data(self, features_df: pd.DataFrame, elements_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for training/testing."""
        # Ensure matching indices
        merged_df = pd.merge(features_df, elements_df, on='spectrum_id')
        
        # Extract features and labels
        X = merged_df.drop(columns=['spectrum_id'] + list(elements_df.columns)[1:])
        y = merged_df[list(elements_df.columns)[1:]]  # All element columns
        
        self.feature_columns = X.columns.tolist()
        self.element_labels = y.columns.tolist()
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2, 
              random_state: int = 42, tune_hyperparameters: bool = False) -> Dict:
        """Train the element prediction model."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Base model
        rf = RandomForestClassifier(random_state=random_state)
        
        # Multi-output wrapper for multi-label classification
        model = MultiOutputClassifier(rf, n_jobs=-1)
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            param_grid = {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': [None, 10, 20, 30],
                'estimator__min_samples_split': [2, 5, 10],
                'estimator__min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results = {}
        for i, element in enumerate(self.element_labels):
            element_acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
            results[element] = element_acc
            
        # Overall accuracy
        overall_acc = accuracy_score(y_test, y_pred)
        results['overall_accuracy'] = overall_acc
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=self.element_labels, output_dict=True)
        results['classification_report'] = report
        
        self.model = model
        
        return {
            'model': model,
            'results': results,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict elements for new data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
            
        # Ensure X has the same columns as training data
        if set(X.columns) != set(self.feature_columns):
            missing_cols = set(self.feature_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(self.feature_columns)
            
            if missing_cols:
                raise ValueError(f"Missing columns in input data: {missing_cols}")
            if extra_cols:
                print(f"Warning: Extra columns in input data will be ignored: {extra_cols}")
                X = X[self.feature_columns]
                
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> List[np.ndarray]:
        """Predict probability of each element for new data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
            
        # Ensure X has the same columns as training data
        if set(X.columns) != set(self.feature_columns):
            missing_cols = set(self.feature_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(self.feature_columns)
            
            if missing_cols:
                raise ValueError(f"Missing columns in input data: {missing_cols}")
            if extra_cols:
                print(f"Warning: Extra columns in input data will be ignored: {extra_cols}")
                X = X[self.feature_columns]
                
        return self.model.predict_proba(X)
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
            
        model_data = {
            'model': self.model,
            'element_labels': self.element_labels,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.element_labels = model_data['element_labels']
        self.feature_columns = model_data['feature_columns']
        
        print(f"Model loaded from {path}")
