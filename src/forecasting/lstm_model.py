"""
LSTM Forecasting Module

Implements LSTM-based time series forecasting for disk storage prediction
with sequence learning and deep learning capabilities.
"""

import warnings
from typing import Tuple, Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Try to import TensorFlow/Keras, fall back gracefully if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes when TensorFlow is not available
    class Sequential:
        pass
    class LSTM:
        pass
    class Dense:
        pass
    print("Warning: TensorFlow not available. LSTM functionality will be limited.")

warnings.filterwarnings('ignore')


class LSTMForecaster:
    """
    LSTM model for time series forecasting.
    
    Attributes:
        model: Compiled Keras LSTM model
        scaler: MinMaxScaler for data normalization
        sequence_length: Length of input sequences
        data: Original time series data
        scaled_data: Normalized data
        predictions: Forecasted values
    """
    
    def __init__(self, sequence_length: int = 60, 
                 lstm_units: List[int] = [50, 50],
                 dropout: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length: Number of time steps to look back
            lstm_units: List of LSTM layer units
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM forecasting. Install with: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        self.predictions = None
        self.history = None
        
    def create_sequences(self, data: np.ndarray, 
                        sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and targets for LSTM training.
        
        Args:
            data: Time series data
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build and compile LSTM model.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()
        
        # First LSTM layer
        if len(self.lstm_units) > 1:
            model.add(LSTM(units=self.lstm_units[0], 
                          return_sequences=True, 
                          input_shape=input_shape))
            model.add(Dropout(self.dropout))
            
            # Additional LSTM layers
            for i in range(1, len(self.lstm_units) - 1):
                model.add(LSTM(units=self.lstm_units[i], 
                              return_sequences=True))
                model.add(Dropout(self.dropout))
            
            # Last LSTM layer
            model.add(LSTM(units=self.lstm_units[-1]))
            model.add(Dropout(self.dropout))
        else:
            # Single LSTM layer
            model.add(LSTM(units=self.lstm_units[0], 
                          input_shape=input_shape))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def prepare_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Convert to numpy array and reshape
        dataset = data.values.reshape(-1, 1)
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(dataset)
        
        # Create sequences
        X, y = self.create_sequences(self.scaled_data, self.sequence_length)
        
        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Reshape X for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, data: pd.Series, 
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.2,
            verbose: int = 1,
            patience: int = 10) -> 'LSTMForecaster':
        """
        Fit LSTM model to time series data.
        
        Args:
            data: Time series data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            verbose: Verbosity level
            patience: Early stopping patience
            
        Returns:
            Fitted LSTMForecaster instance
        """
        print(f"Preparing LSTM data with sequence length: {self.sequence_length}")
        
        self.data = data
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        print(f"LSTM Model Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        # Train model
        print("Training LSTM model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[early_stop, reduce_lr]
        )
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        print("LSTM model training completed!")
        
        return self
    
    def predict(self, steps: int, 
                input_data: Optional[pd.Series] = None) -> np.ndarray:
        """
        Generate forecasts for future time periods.
        
        Args:
            steps: Number of steps to forecast
            input_data: Optional input data, uses training data if None
            
        Returns:
            Array of forecasted values (original scale)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        if input_data is not None:
            # Use provided data
            data_scaled = self.scaler.transform(input_data.values.reshape(-1, 1))
            last_sequence = data_scaled[-self.sequence_length:]
        else:
            # Use last sequence from training data
            last_sequence = self.scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            X_pred = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Make prediction
            pred_scaled = self.model.predict(X_pred, verbose=0)
            predictions.append(pred_scaled[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_scaled
        
        # Convert back to original scale
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_unscaled = self.scaler.inverse_transform(predictions_array)
        
        self.predictions = predictions_unscaled.flatten()
        
        return self.predictions
    
    def evaluate(self, test_data: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            # Use stored test data from training
            y_pred_scaled = self.model.predict(self.X_test, verbose=0)
            
            # Convert back to original scale
            y_test_unscaled = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
            y_pred_unscaled = self.scaler.inverse_transform(y_pred_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
            rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
            mape = np.mean(np.abs((y_test_unscaled - y_pred_unscaled) / y_test_unscaled)) * 100
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'test_samples': len(y_test_unscaled)
            }
            
            return metrics
        else:
            raise ValueError("No test data available for evaluation")
    
    def plot_training_history(self) -> plt.Figure:
        """
        Plot training history.
        
        Returns:
            Matplotlib figure
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary and configuration.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not fitted"}
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        summary = {
            'architecture': 'LSTM',
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'optimizer': 'Adam',
            'loss_function': 'MSE'
        }
        
        if self.history:
            summary['epochs_trained'] = len(self.history.history['loss'])
            summary['final_train_loss'] = self.history.history['loss'][-1]
            summary['final_val_loss'] = self.history.history['val_loss'][-1]
        
        return summary


def prepare_data_for_lstm(df: pd.DataFrame, 
                         date_col: str = 'Date',
                         value_col: str = 'Usage') -> pd.Series:
    """
    Prepare DataFrame for LSTM forecasting.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        value_col: Name of value column
        
    Returns:
        Prepared time series
    """
    df_clean = df.copy()
    
    # Convert date column
    if date_col in df_clean.columns:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_clean.set_index(date_col, inplace=True)
    
    # Sort by date
    df_clean.sort_index(inplace=True)
    
    # Handle missing values
    df_clean[value_col] = df_clean[value_col].interpolate(method='linear')
    
    return df_clean[value_col]


class LSTMEnsemble:
    """
    Ensemble of LSTM models for improved forecasting accuracy.
    """
    
    def __init__(self, n_models: int = 3):
        """
        Initialize LSTM ensemble.
        
        Args:
            n_models: Number of models in ensemble
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM forecasting")
            
        self.n_models = n_models
        self.models = []
        self.fitted = False
    
    def fit(self, data: pd.Series, **kwargs) -> 'LSTMEnsemble':
        """
        Fit ensemble of LSTM models.
        
        Args:
            data: Time series data
            **kwargs: Arguments passed to individual models
            
        Returns:
            Fitted LSTMEnsemble instance
        """
        self.models = []
        
        for i in range(self.n_models):
            print(f"\nTraining LSTM model {i+1}/{self.n_models}")
            
            # Create model with slight variations
            lstm_units = [50 + i*10, 50 + i*5] if i < 2 else [60, 40]
            sequence_length = 60 + i*10
            
            model = LSTMForecaster(
                sequence_length=sequence_length,
                lstm_units=lstm_units,
                dropout=0.2 + i*0.05
            )
            
            model.fit(data, verbose=0, **kwargs)
            self.models.append(model)
        
        self.fitted = True
        return self
    
    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        all_predictions = []
        
        for model in self.models:
            pred = model.predict(steps)
            all_predictions.append(pred)
        
        predictions_array = np.array(all_predictions)
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        return mean_pred, std_pred