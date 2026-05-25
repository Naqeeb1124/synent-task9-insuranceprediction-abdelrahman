import os
import joblib
import pandas as pd
import numpy as np

class InsurancePredictor:
    def __init__(self, model_path=os.path.join("models", "best_model.pkl")):
        self.model_path = model_path
        self.model_payload = None
        self.load_model()
        
    def load_model(self):
        """
        Loads the trained model payload.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model payload not found at '{self.model_path}'. "
                "Please run model training first: 'python src/train.py'"
            )
        self.model_payload = joblib.load(self.model_path)
        print(f"Loaded {self.model_payload['model_name']} model.")
        
    def predict(self, input_dict):
        """
        Predicts insurance charges for a single user profile.
        input_dict format:
        {
            'age': 30,
            'sex': 'male',
            'bmi': 25.4,
            'children': 2,
            'smoker': 'yes',
            'region': 'southwest'
        }
        """
        if self.model_payload is None:
            self.load_model()
            
        # Convert dictionary to DataFrame
        df_input = pd.DataFrame([input_dict])
        
        # Preprocess
        preprocessor = self.model_payload['preprocessor']
        X_processed = preprocessor.transform(df_input)
        
        # Predict
        model = self.model_payload['model']
        prediction = model.predict(X_processed)[0]
        
        return round(float(prediction), 2)
    
    def analyze_cost_drivers(self, input_dict):
        """
        Performs Ceteris Paribus (marginal impact analysis) to determine
        how much each factor (Smoking, High BMI, Age) adds to the premium
        relative to a baseline healthy profile.
        
        Baseline Profile:
        - Age: 18 (Youngest adult)
        - Smoker: 'no' (Non-smoker)
        - BMI: 22.0 (Normal weight range)
        - Children: 0 (No dependents)
        - Sex: Same as user (to isolate behavior factors)
        - Region: Same as user
        """
        if self.model_payload is None:
            self.load_model()
            
        # Baseline dictionary
        baseline_dict = {
            'age': 18,
            'sex': input_dict['sex'],
            'bmi': 22.0,
            'children': 0,
            'smoker': 'no',
            'region': input_dict['region']
        }
        
        # 1. Base Premium (standard profile)
        base_premium = self.predict(baseline_dict)
        
        # 2. Age Impact: user's age, other factors baseline
        age_only = baseline_dict.copy()
        age_only['age'] = input_dict['age']
        age_premium = self.predict(age_only)
        age_impact = max(0.0, age_premium - base_premium)
        
        # 3. Smoker Impact: user's smoker status, other factors user's baseline
        # (We evaluate this at the user's current age/BMI to capture interactions)
        non_smoker_profile = input_dict.copy()
        non_smoker_profile['smoker'] = 'no'
        non_smoker_premium = self.predict(non_smoker_profile)
        user_premium = self.predict(input_dict)
        smoker_impact = max(0.0, user_premium - non_smoker_premium) if input_dict['smoker'] == 'yes' else 0.0
        
        # 4. BMI Impact: user's BMI vs baseline BMI, other factors user's baseline
        normal_bmi_profile = input_dict.copy()
        normal_bmi_profile['bmi'] = 22.0
        normal_bmi_premium = self.predict(normal_bmi_profile)
        bmi_impact = max(0.0, user_premium - normal_bmi_premium) if input_dict['bmi'] > 24.9 else 0.0
        
        # 5. Dependents/Children Impact
        no_children_profile = input_dict.copy()
        no_children_profile['children'] = 0
        no_children_premium = self.predict(no_children_profile)
        children_impact = max(0.0, user_premium - no_children_premium) if input_dict['children'] > 0 else 0.0
        
        # 6. Flat charges or base coverage (everything else)
        # We define base coverage as the minimum charge or the non-smoker normal BMI age 18 charge.
        base_coverage = base_premium
        
        drivers = {
            'Base Coverage': round(base_coverage, 2),
            'Age Factor': round(age_impact, 2),
            'Smoking Surcharge': round(smoker_impact, 2),
            'BMI Surcharge': round(bmi_impact, 2),
            'Dependents Surcharge': round(children_impact, 2),
            'Total Estimated Premium': round(user_premium, 2)
        }
        
        return drivers

if __name__ == "__main__":
    # Test predictor if model exists
    try:
        predictor = InsurancePredictor()
        test_profile = {
            'age': 35,
            'sex': 'male',
            'bmi': 28.5,
            'children': 1,
            'smoker': 'yes',
            'region': 'southwest'
        }
        pred = predictor.predict(test_profile)
        drivers = predictor.analyze_cost_drivers(test_profile)
        print(f"Test profile prediction: ${pred}")
        print("Cost driver analysis:")
        for k, v in drivers.items():
            print(f" - {k}: ${v}")
    except Exception as e:
        print(f"Model test skipped or failed: {e}")
