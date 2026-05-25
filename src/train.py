import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import custom modules
from data_preprocessing import clean_data, build_preprocessor

def train_and_evaluate():
    print("Loading cleaned dataset...")
    cleaned_path = os.path.join("data", "processed", "cleaned_insurance.csv")
    
    if not os.path.exists(cleaned_path):
        print("Processed data not found. Running preprocessing pipeline first...")
        from data_preprocessing import run_preprocessing_pipeline
        df = run_preprocessing_pipeline()
    else:
        df = pd.read_csv(cleaned_path)
        
    # Split features and target
    X = df.drop(columns=['charges'])
    y = df['charges']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split: Train shape = {X_train.shape}, Test shape = {X_test.shape}")
    
    # Build and fit preprocessing pipeline
    print("Fitting preprocessing pipeline...")
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding for later analysis
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_features = preprocessor.transformers_[1][2]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    num_features = preprocessor.transformers_[0][2]
    feature_names = num_features + encoded_cat_names
    
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=4, learning_rate=0.08)
    }
    
    results = {}
    trained_models = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_processed, y_train)
        
        # Predict
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        results[name] = {
            "R2 Score": r2,
            "MAE": mae,
            "RMSE": rmse
        }
        trained_models[name] = model
        
        print(f"{name} Results: R2 = {r2:.4f} | MAE = ${mae:.2f} | RMSE = ${rmse:.2f}")
        
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    print("\n--- Model Performance Comparison ---")
    print(results_df.to_string())
    
    # Select best model based on R2 Score
    best_model_name = results_df["R2 Score"].idxmax()
    best_model = trained_models[best_model_name]
    best_r2 = results[best_model_name]["R2 Score"]
    
    print(f"\nBest Model: {best_model_name} (R2 = {best_r2:.4f})")
    
    # Extract feature importances if available
    feature_importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importances = dict(zip(feature_names, importances.tolist()))
        # Sort importances
        feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
        print("\nFeature Importances (Best Model):")
        for feat, imp in list(feature_importances.items())[:5]:
            print(f" - {feat}: {imp:.4f}")
            
    # Save the best model, preprocessor, features list, and importances together as a single artifact
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    payload_path = os.path.join(models_dir, "best_model.pkl")
    
    payload = {
        "model_name": best_model_name,
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "metrics": results[best_model_name],
        "comparison": results
    }
    
    joblib.dump(payload, payload_path)
    print(f"\nSaved best model and preprocessor payload successfully to {payload_path}")

if __name__ == "__main__":
    train_and_evaluate()
