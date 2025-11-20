"""
Healthcare Cost Prediction using Deep Learning
Based on: "Deep learning for prediction of population health costs"
BMC Medical Informatics and Decision Making (2022)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: DATA PREPARATION & FEATURE ENGINEERING
# ============================================================================

class HealthCostDataPreprocessor:
    """
    Preprocesses health insurance claims data following the paper's methodology
    """
    
    def __init__(self, n_quarters=8):
        """
        Args:
            n_quarters: Number of quarters to use for observation (paper used 24)
        """
        self.n_quarters = n_quarters
        self.categorical_encoders = {}
        self.feature_names = []
        self.feature_dim = 0
        
    def create_quarterly_vectors(self, df):
        """
        Creates quarterly aggregated vectors for each patient
        Following paper's approach: one-hot encode categorical variables,
        aggregate by quarters, and concatenate into single vector
        """
        print("Creating quarterly feature vectors...")
        
        # Group by patient and quarter
        df['Quarter'] = pd.to_datetime(df['Paid Date']).dt.to_period('Q')
        
        # Define categorical columns for one-hot encoding
        categorical_cols = [
            'Condition (Impairment Code)',
            'Treatment Type',
            'Claim Type',
            'Treatment Provider Type',
            'Gender'
        ]
        
        # Define numerical columns
        numerical_cols = ['Claim Amount', 'Amount Paid', 'Age']
        
        patient_vectors = []
        patient_ids = []
        target_costs = []
        
        # Process each patient
        for patient_id in df['Claimant Unique ID'].unique():
            patient_data = df[df['Claimant Unique ID'] == patient_id].copy()
            
            # Create vector for each quarter
            quarterly_vectors = []
            
            for quarter in range(self.n_quarters):
                quarter_data = patient_data[patient_data['Quarter'].astype(str) == 
                                          patient_data['Quarter'].unique()[0] if quarter < len(patient_data['Quarter'].unique()) else None]
                
                # Initialize quarter vector
                quarter_vector = []
                
                # Process categorical features (one-hot encoded and summed if multiple)
                for col in categorical_cols:
                    if col not in self.categorical_encoders:
                        # Fit encoder on entire dataset
                        self.categorical_encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        self.categorical_encoders[col].fit(df[[col]])
                    
                    if len(quarter_data) > 0:
                        # One-hot encode and sum if multiple entries
                        encoded = self.categorical_encoders[col].transform(quarter_data[[col]])
                        summed = encoded.sum(axis=0)
                        quarter_vector.extend(summed)
                    else:
                        # Zero vector if no data for this quarter
                        n_categories = len(self.categorical_encoders[col].categories_[0])
                        quarter_vector.extend([0] * n_categories)
                
                # Process numerical features (sum or mean)
                for col in numerical_cols:
                    if len(quarter_data) > 0:
                        quarter_vector.append(quarter_data[col].sum())
                    else:
                        quarter_vector.append(0)
                
                quarterly_vectors.extend(quarter_vector)
            
            patient_vectors.append(quarterly_vectors)
            patient_ids.append(patient_id)
            
            # Target: sum of costs in next 12 months (last available amount)
            target_costs.append(patient_data['Amount Paid'].sum())
        
        self.feature_dim = len(quarterly_vectors)
        print(f"Feature dimension per patient: {self.feature_dim}")
        
        return np.array(patient_vectors), np.array(target_costs), patient_ids
    
    def prepare_data(self, df, test_size=0.3):
        """
        Complete data preparation pipeline
        """
        # Create feature vectors
        X, y, patient_ids = self.create_quarterly_vectors(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test

# ============================================================================
# STEP 2: DEEP NEURAL NETWORK MODEL (Following Paper Architecture)
# ============================================================================

class DeepCostPredictor:
    """
    Deep Neural Network for cost prediction
    Architecture from paper:
    - 4 hidden layers with 50 neurons each
    - ReLU activation
    - Dropout 0.25
    - Skip connection: concatenate input to last hidden layer
    """
    
    def __init__(self, input_dim, n_cost_categories=7):
        """
        Args:
            input_dim: Dimension of input features
            n_cost_categories: Number of cost categories to predict (paper used 7)
        """
        self.input_dim = input_dim
        self.n_cost_categories = n_cost_categories
        self.model = None
        
    def build_model(self):
        """
        Build the deep neural network following paper's architecture
        """
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,), name='input')
        
        # First hidden layer
        x = layers.Dense(50, activation='relu', name='hidden_1')(input_layer)
        x = layers.Dropout(0.25, name='dropout_1')(x)
        
        # Second hidden layer
        x = layers.Dense(50, activation='relu', name='hidden_2')(x)
        x = layers.Dropout(0.25, name='dropout_2')(x)
        
        # Third hidden layer
        x = layers.Dense(50, activation='relu', name='hidden_3')(x)
        x = layers.Dropout(0.25, name='dropout_3')(x)
        
        # Fourth hidden layer
        x = layers.Dense(50, activation='relu', name='hidden_4')(x)
        x = layers.Dropout(0.25, name='dropout_4')(x)
        
        # Skip connection: concatenate original input
        x = layers.Concatenate(name='skip_connection')([x, input_layer])
        
        # Output layer - predict total cost (simplified to 1 output instead of 7 categories)
        output = layers.Dense(1, activation='linear', name='output')(x)
        
        # Create model
        self.model = models.Model(inputs=input_layer, outputs=output)
        
        # Compile with ADAM optimizer (paper settings)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print("\nModel Architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=25, batch_size=32):
        """
        Train the model (paper used 25 epochs, batch size 32)
        """
        print("\nTraining model...")
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance with metrics from paper
        """
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Pearson correlation
        pearson_corr = np.corrcoef(y_test, predictions)[0, 1]
        
        # Spearman correlation (rank-based)
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(y_test, predictions)
        
        results = {
            'MAE': mae,
            'R2': r2,
            'Pearson Correlation': pearson_corr,
            'Spearman Correlation': spearman_corr
        }
        
        return results, predictions

# ============================================================================
# STEP 3: RIDGE REGRESSION BASELINE (for comparison)
# ============================================================================

from sklearn.linear_model import Ridge

class RidgeRegressionBaseline:
    """
    Ridge Regression baseline for comparison (paper used lambda=0.1)
    """
    
    def __init__(self, alpha=0.1):
        self.model = Ridge(alpha=alpha)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        pearson_corr = np.corrcoef(y_test, predictions)[0, 1]
        
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(y_test, predictions)
        
        results = {
            'MAE': mae,
            'R2': r2,
            'Pearson Correlation': pearson_corr,
            'Spearman Correlation': spearman_corr
        }
        
        return results, predictions

# ============================================================================
# STEP 4: MAIN EXECUTION PIPELINE
# ============================================================================

def create_sample_data():
    """
    Create sample dataset based on the images provided
    """
    data = {
        'Claimant Unique ID': ['EMP00000001-01'] * 5 + ['EMP00000003-01'] * 5 + ['EMP00000003-02'] * 3,
        'Unique Medical Claim ID': ['CLM00028'] * 5 + ['CLM00094'] * 5 + ['CLM00011'] * 3,
        'Incurred Date': pd.date_range('2023-01-01', periods=13, freq='M'),
        'Paid Date': pd.date_range('2023-01-15', periods=13, freq='M'),
        'Condition (Impairment Code)': ['M23.2', 'M75.1', 'M17.0', 'M23.2', 'M25.5'] + 
                                        ['M51.2', 'M23.2', 'M23.2', 'M25.5', 'M51.2'] + 
                                        ['M54.5', 'M25.5', 'M75.1'],
        'Treatment Type': ['Surgical', 'Therapeutic', 'Diagnostic', 'Medical', 'Diagnostic'] * 2 + ['Therapeutic'] * 3,
        'Claim Type': ['Day Case', 'Day Case', 'Outpatient', 'Outpatient', 'Outpatient'] * 2 + ['Day Case'] * 3,
        'Treatment Provider Type': ['Hospital', 'Hospital', 'Clinic', 'Hospital', 'Diagnostic Center'] * 2 + ['Clinic'] * 3,
        'Claim Amount': [7029.06, 4876.84, 2650.31, 1202.24, 5289.99, 
                         4260.42, 3257.69, 1696.22, 1079.18, 7398.98,
                         2826.91, 12881.86, 4483.48],
        'Amount Paid': [5964.28, 3910.81, 2641.74, 1030.53, 4584.14,
                        3617.47, 3126.90, 1646.88, 993.92, 6609.69,
                        2660.59, 10943.36, 3636.66],
        'Gender': ['Female'] * 5 + ['Female'] * 5 + ['Male'] * 3,
        'Age': [44] * 5 + [44] * 5 + [72] * 3,
        'Year of Birth': [1978] * 10 + [1947] * 3
    }
    
    return pd.DataFrame(data)

def main():
    """
    Main execution pipeline
    """
    print("="*80)
    print("HEALTHCARE COST PREDICTION USING DEEP LEARNING")
    print("Based on: BMC Medical Informatics and Decision Making (2022)")
    print("="*80)
    
    # Load sample data
    print("\n1. Loading sample data...")
    df = create_sample_data()
    print(f"Loaded {len(df)} claims records")
    print(f"Unique patients: {df['Claimant Unique ID'].nunique()}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = HealthCostDataPreprocessor(n_quarters=4)  # Using 4 quarters for demo
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Build and train Deep Neural Network
    print("\n3. Building Deep Neural Network...")
    dnn_model = DeepCostPredictor(input_dim=X_train.shape[1])
    dnn_model.build_model()
    
    print("\n4. Training Deep Neural Network...")
    history = dnn_model.train(X_train, y_train, X_test, y_test, epochs=25, batch_size=32)
    
    # Evaluate DNN
    print("\n5. Evaluating Deep Neural Network...")
    dnn_results, dnn_predictions = dnn_model.evaluate(X_test, y_test)
    
    print("\n" + "="*80)
    print("DEEP NEURAL NETWORK RESULTS:")
    print("="*80)
    for metric, value in dnn_results.items():
        print(f"{metric:25s}: {value:.4f}")
    
    # Train Ridge Regression baseline
    print("\n6. Training Ridge Regression Baseline...")
    ridge_model = RidgeRegressionBaseline(alpha=0.1)
    ridge_model.train(X_train, y_train)
    
    # Evaluate Ridge
    print("\n7. Evaluating Ridge Regression...")
    ridge_results, ridge_predictions = ridge_model.evaluate(X_test, y_test)
    
    print("\n" + "="*80)
    print("RIDGE REGRESSION RESULTS:")
    print("="*80)
    for metric, value in ridge_results.items():
        print(f"{metric:25s}: {value:.4f}")
    
    # Comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON:")
    print("="*80)
    print(f"{'Metric':<25s} {'Deep NN':>12s} {'Ridge':>12s} {'Winner':>12s}")
    print("-"*80)
    for metric in dnn_results.keys():
        dnn_val = dnn_results[metric]
        ridge_val = ridge_results[metric]
        if metric == 'MAE':
            winner = "Deep NN" if dnn_val < ridge_val else "Ridge"
        else:
            winner = "Deep NN" if dnn_val > ridge_val else "Ridge"
        print(f"{metric:<25s} {dnn_val:>12.4f} {ridge_val:>12.4f} {winner:>12s}")
    
    # Sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (First 5 test samples):")
    print("="*80)
    print(f"{'Actual':<15s} {'DNN Predicted':<15s} {'Ridge Predicted':<15s} {'DNN Error':<15s} {'Ridge Error':<15s}")
    print("-"*80)
    for i in range(min(5, len(y_test))):
        actual = y_test[i]
        dnn_pred = dnn_predictions[i]
        ridge_pred = ridge_predictions[i]
        dnn_error = abs(actual - dnn_pred)
        ridge_error = abs(actual - ridge_pred)
        print(f"{actual:<15.2f} {dnn_pred:<15.2f} {ridge_pred:<15.2f} {dnn_error:<15.2f} {ridge_error:<15.2f}")
    
    print("\n" + "="*80)
    print("NEXT STEPS FOR YOUR DATA:")
    print("="*80)
    print("1. Replace create_sample_data() with your actual data loading")
    print("2. Adjust n_quarters based on your historical data (paper used 24)")
    print("3. Add more cost categories if needed (paper predicted 7 categories)")
    print("4. Implement model ensembling (train 5 models and average predictions)")
    print("5. Add integrated gradients for feature importance analysis")
    print("6. Scale up with more training data for better performance")
    print("="*80)

if __name__ == "__main__":
    main()
