# src/models/molecule_predictor.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
import joblib
from typing import Dict, List, Tuple, Optional

class MoleculePredictor:
    """Predict molecules from mass spectrometry data using deep learning."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.molecule_labels = None
        self.feature_columns = None
        self.mlb = None
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_data(self, features_df: pd.DataFrame, molecules_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data for training/testing."""
        # Ensure matching indices
        merged_df = pd.merge(features_df, molecules_df, on='spectrum_id')
        
        # Extract features
        X = merged_df.drop(columns=['spectrum_id'] + list(molecules_df.columns)[1:])
        self.feature_columns = X.columns.tolist()
        
        # Process molecule labels
        molecules = []
        for _, row in merged_df.iterrows():
            # Get all molecules for this spectrum
            spec_molecules = [col for col in molecules_df.columns[1:] if row[col] == 1]
            molecules.append(spec_molecules)
        
        # Multi-label binarization
        self.mlb = MultiLabelBinarizer()
        y = self.mlb.fit_transform(molecules)
        self.molecule_labels = self.mlb.classes_
        
        return X, y
    
    def build_cnn_model(self, input_shape: Tuple[int, ...], num_classes: int) -> Model:
        """Build a CNN model for molecule prediction."""
        # Separate inputs for different feature types
        binned_input = Input(shape=(input_shape[0],), name='binned_input')
        fp_input = Input(shape=(input_shape[1],), name='fp_input')
        other_input = Input(shape=(input_shape[2],), name='other_input')
        
        # Reshape for CNN
        binned_reshaped = tf.reshape(binned_input, (-1, input_shape[0], 1))
        fp_reshaped = tf.reshape(fp_input, (-1, input_shape[1], 1))
        
        # CNN for binned spectrum
        cnn_binned = Conv1D(32, kernel_size=5, activation='relu')(binned_reshaped)
        cnn_binned = MaxPooling1D(pool_size=2)(cnn_binned)
        cnn_binned = Conv1D(64, kernel_size=3, activation='relu')(cnn_binned)
        cnn_binned = MaxPooling1D(pool_size=2)(cnn_binned)
        cnn_binned = Flatten()(cnn_binned)
        
        # CNN for fingerprint
        cnn_fp = Conv1D(16, kernel_size=3, activation='relu')(fp_reshaped)
        cnn_fp = MaxPooling1D(pool_size=2)(cnn_fp)
        cnn_fp = Conv1D(32, kernel_size=3, activation='relu')(cnn_fp)
        cnn_fp = MaxPooling1D(pool_size=2)(cnn_fp)
        cnn_fp = Flatten()(cnn_fp)
        
        # Concatenate all features
        concatenated = Concatenate()([cnn_binned, cnn_fp, other_input])
        
        # Dense layers
        x = Dense(256, activation='relu')(concatenated)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(num_classes, activation='sigmoid')(x)
        
        # Create model
        model = Model(
            inputs=[binned_input, fp_input, other_input],
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_model_inputs(self, X: pd.DataFrame) -> List[np.ndarray]:
        """Prepare inputs for the model based on feature types."""
        # Extract different feature types
        binned_cols = [col for col in X.columns if col.startswith('binned_')]
        fp_cols = [col for col in X.columns if col.startswith('fp_')]
        other_cols = [col for col in X.columns if col not in binned_cols and col not in fp_cols]
        
        # Create input arrays
        binned_input = X[binned_cols].values
        fp_input = X[fp_cols].values
        other_input = X[other_cols].values
        
        return [binned_input, fp_input, other_input]
    
    def train(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, 
              random_state: int = 42, epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train the molecule prediction model."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Prepare model inputs
        X_train_inputs = self.prepare_model_inputs(X_train)
        X_test_inputs = self.prepare_model_inputs(X_test)
        
        # Build model
        input_shapes = (
            len([col for col in X.columns if col.startswith('binned_')]),
            len([col for col in X.columns if col.startswith('fp_')]),
            len([col for col in X.columns if not col.startswith('binned_') and not col.startswith('fp_')])
        )
        
        model = self.build_cnn_model(input_shapes, y.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('results/models/molecule_predictor_best.h5', save_best_only=True)
        ]
        
        # Train model
        history = model.fit(
            X_train_inputs, y_train,
            validation_data=(X_test_inputs, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test_inputs)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        results = {}
        for i, molecule in enumerate(self.molecule_labels):
            molecule_acc = accuracy_score(y_test[:, i], y_pred[:, i])
            results[molecule] = molecule_acc
            
        # Overall accuracy
        overall_acc = accuracy_score(y_test, y_pred)
        results['overall_accuracy'] = overall_acc
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=self.molecule_labels, 
            output_dict=True,
            zero_division=0
        )
        results['classification_report'] = report
        
        self.model = model
        
        return {
            'model': model,
            'history': history,
            'results': results,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict molecules for new data."""
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
        
        # Prepare inputs
        X_inputs = self.prepare_model_inputs(X)
        
        # Predict
        y_pred_proba = self.model.predict(X_inputs)
        y_pred = (y_pred_proba > threshold).astype(int)
        
        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of each molecule for new data."""
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
        
        # Prepare inputs
        X_inputs = self.prepare_model_inputs(X)
        
        # Predict probabilities
        y_pred_proba = self.model.predict(X_inputs)
        
        return y_pred_proba
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
            
        # Save Keras model
        self.model.save(path.replace('.h5', '_model.h5'))
        
        # Save additional data
        model_data = {
            'molecule_labels': self.molecule_labels,
            'feature_columns': self.feature_columns,
            'mlb': self.mlb
        }
        
        joblib.dump(model_data, path.replace('.h5', '_data.pkl'))
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        # Load Keras model
        self.model = tf.keras.models.load_model(path.replace('.h5', '_model.h5'))
        
        # Load additional data
        model_data = joblib.load(path.replace('.h5', '_data.pkl'))
        
        self.molecule_labels = model_data['molecule_labels']
        self.feature_columns = model_data['feature_columns']
        self.mlb = model_data['mlb']
        
        print(f"Model loaded from {path}")
