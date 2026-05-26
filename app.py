import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import sys

# Set page configuration
st.set_page_config(
    page_title="HealthSure Insurance Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to load data science model and artifacts
@st.cache_resource
def load_predictor():
    model_path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(model_path):
        # Auto-train if model is missing (useful for cloud deployments)
        with st.status("Initializing Machine Learning Engine...", expanded=True) as status:
            try:
                # 1. Ensure raw data exists
                raw_data_path = os.path.join("data", "raw", "insurance.csv")
                if not os.path.exists(raw_data_path):
                    st.write("Downloading raw dataset...")
                    subprocess.run([sys.executable, "src/download_data.py"], check=True)
                
                # 2. Run training (which includes preprocessing)
                st.write("Training models and optimizing performance...")
                subprocess.run([sys.executable, "src/train.py"], check=True)
                
                status.update(label="Engine ready!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Auto-training failed: {e}")
                return None
                
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# Helper to load processed dataset for visualization
@st.cache_data
def load_visualization_data():
    csv_path = os.path.join("data", "processed", "cleaned_insurance.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)

# Load model and data
payload = load_predictor()
df_visual = load_visualization_data()

# ----------------- HEADER -----------------
st.title("HealthSure Insurance Predictor")
st.write("Advanced Predictive Risk Analytics & Machine Learning-Driven Premium Estimator")

if payload is None:
    st.error("The Machine Learning model has not been trained yet. Please run `python src/train.py` in the workspace terminal to build the models.")
    st.stop()

# Extract model metrics & comparisons
best_model_name = payload['model_name']
model_comparison = payload['comparison']
feature_importances = payload['feature_importances']

# ----------------- TABS CREATION -----------------
tab1, tab2, tab3 = st.tabs([
    "Premium Calculator", 
    "Exploratory Data Analysis", 
    "Model Analytics"
])

# ==============================================================================
# TAB 1: PREMIUM CALCULATOR
# ==============================================================================
with tab1:
    st.header("Calculate Your Personalized Policy Estimate")
    st.write("Modify the demographics and health parameters in the sections below to dynamically estimate your annual medical insurance premium.")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Demographics & Profile")
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            age = st.slider("Select Age", min_value=18, max_value=100, value=30, step=1)
        with col_a2:
            sex = st.selectbox("Select Gender", options=['Female', 'Male'], index=0)
            
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            children = st.number_input("Number of Dependents (Children)", min_value=0, max_value=10, value=1, step=1)
        with col_b2:
            region = st.selectbox("Residential Region", options=['Southwest', 'Southeast', 'Northwest', 'Northeast'], index=0)
            
        st.divider()
        st.subheader("Health & Lifestyle Risk Factors")
        
        smoker = st.radio("Do you smoke regularly?", options=['No', 'Yes'], index=0, horizontal=True)
        
        st.write("**Body Mass Index (BMI) Settings**")
        bmi_mode = st.radio("BMI Input Mode", options=["Standard BMI Slider", "Calculate BMI (Height/Weight)"], horizontal=True)
        
        if bmi_mode == "Standard BMI Slider":
            bmi = st.slider("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        else:
            col_bmi1, col_bmi2 = st.columns(2)
            with col_bmi1:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.5)
            with col_bmi2:
                height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0, step=1.0)
            
            height_m = height_cm / 100.0
            bmi = round(weight / (height_m ** 2), 1)
            
            if bmi < 18.5:
                st.info(f"Calculated BMI: {bmi} (Underweight)")
            elif 18.5 <= bmi < 25:
                st.success(f"Calculated BMI: {bmi} (Healthy Weight)")
            elif 25 <= bmi < 30:
                st.warning(f"Calculated BMI: {bmi} (Overweight)")
            else:
                st.error(f"Calculated BMI: {bmi} (Obese)")

    with col2:
        input_data = {
            'age': age,
            'sex': sex.lower(),
            'bmi': bmi,
            'children': children,
            'smoker': smoker.lower(),
            'region': region.lower()
        }
        
        # Load Predictor engine
        from src.predict import InsurancePredictor
        predictor = InsurancePredictor()
        
        # Calculate
        predicted_charges = predictor.predict(input_data)
        drivers = predictor.analyze_cost_drivers(input_data)
        
        st.metric(
            label="Estimated Annual Premium", 
            value=f"${predicted_charges:,.2f}",
            delta=f"Powered by {best_model_name}",
            delta_color="off"
        )
        
        st.subheader("Premium Composition Breakdown")
        
        # Construct Plotly horizontal Bar chart (default styling)
        driver_keys = ['Base Coverage', 'Age Factor', 'Smoking Surcharge', 'BMI Surcharge', 'Dependents Surcharge']
        driver_values = [drivers[k] for k in driver_keys]
        
        fig_breakdown = go.Figure(go.Bar(
            x=driver_values,
            y=driver_keys,
            orientation='h'
        ))
        fig_breakdown.update_layout(
            margin=dict(l=0, r=20, t=10, b=10),
            height=240,
            xaxis=dict(title='Contribution ($ USD)', tickformat='$'),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)
        
        st.subheader("Actuarial Risk Assessment Insights")
        
        if smoker == "Yes":
            st.warning(f"Smoking Surcharge: Smoking regularly adds an estimated ${drivers['Smoking Surcharge']:,.2f} to your premium.")
        if bmi > 24.9:
            st.warning(f"High BMI Warning: Your BMI of {bmi} is outside the healthy range, contributing ${drivers['BMI Surcharge']:,.2f} annually.")
        if age > 45:
            st.info(f"Age Bracket Adjustment: Age contributes ${drivers['Age Factor']:,.2f} to the premium compared to a young adult.")
        if children > 0:
            st.info(f"Dependents Surcharge: Adding dependents incurs an adjustment of ${drivers['Dependents Surcharge']:,.2f}.")

# ==============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab2:
    st.header("Interactive Exploratory Data Analysis")
    st.write("Explore raw demographic correlations and risk charge layers within the historical insurance client base.")
    
    if df_visual is None:
        st.info("No processed dataset found. Running pipeline...")
    else:
        # Layout metrics overview of the population using native st.metric
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Population Samples", f"{len(df_visual):,}")
        col_m2.metric("Average Premium Charge", f"${df_visual['charges'].mean():,.2f}")
        col_m3.metric("Smoker Ratio", f"{(df_visual['smoker'] == 'yes').mean()*100:.1f}%")
        col_m4.metric("Average Body Mass Index", f"{df_visual['bmi'].mean():.1f} kg/m²")
            
        st.divider()
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Actuarial Charges Distribution")
            fig_hist = px.histogram(
                df_visual, 
                x="charges", 
                nbins=50, 
                marginal="rug",
                title="Premium Billed Bins ($ USD)"
            )
            fig_hist.update_layout(xaxis=dict(tickformat='$'))
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_g2:
            st.subheader("Lifestyle Impact: Smoker vs Charges")
            fig_box = px.box(
                df_visual, 
                x="smoker", 
                y="charges", 
                color="smoker",
                title="Premium Range Billed by Smoker Profile"
            )
            fig_box.update_layout(yaxis=dict(tickformat='$'))
            st.plotly_chart(fig_box, use_container_width=True)
            
        st.divider()
        st.subheader("Risk Demographics Cluster Map")
        st.write("This scatter plot visualizes the three clean risk 'stripes' or layers present in medical bills.")
        
        fig_scatter = px.scatter(
            df_visual, 
            x="age", 
            y="charges", 
            color="smoker", 
            size="bmi", 
            hover_data=['sex', 'children', 'region'],
            title="Premium Billed vs. Age (Size mapped to BMI, Color mapped to Smoker Status)"
        )
        fig_scatter.update_layout(
            xaxis=dict(title='Age'),
            yaxis=dict(title='Premium Billed ($ USD)', tickformat='$'),
            height=500
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# TAB 3: MODEL ARENA & ANALYTICS
# ==============================================================================
with tab3:
    st.header("Model Arena & Decision Architecture")
    st.write("Compare results from the various candidate machine learning models evaluated during compilation.")
    
    col_c1, col_c2 = st.columns([1, 1], gap="large")
    
    with col_c1:
        st.subheader("Evaluation Metrics Comparison")
        
        comp_df = pd.DataFrame(model_comparison).T
        
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(
            name='R² Score (Accuracy)',
            x=comp_df.index,
            y=comp_df['R2 Score']
        ))
        fig_metrics.update_layout(
            yaxis=dict(title='R² Score', range=[0, 1.0]),
            height=300,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        st.write("**Error Performance Analysis:**")
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Bar(
            name='Mean Absolute Error (MAE)',
            x=comp_df.index,
            y=comp_df['MAE']
        ))
        fig_errors.add_trace(go.Bar(
            name='Root Mean Square Error (RMSE)',
            x=comp_df.index,
            y=comp_df['RMSE']
        ))
        fig_errors.update_layout(
            yaxis=dict(title='Error ($ USD)', tickformat='$'),
            height=300,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_errors, use_container_width=True)

    with col_c2:
        st.subheader(f"Decision Drivers: {best_model_name}")
        st.write("Machine learning decisions are dominated by smoking habits and physical health indicators (BMI and Age).")
        
        if feature_importances is not None:
            feat_df = pd.DataFrame({
                'Feature': list(feature_importances.keys()),
                'Importance': list(feature_importances.values())
            }).sort_values(by='Importance', ascending=True)
            
            fig_importance = px.bar(
                feat_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                title="Model Feature Weight Distribution"
            )
            fig_importance.update_layout(
                xaxis=dict(title='Relative Importance'),
                yaxis=dict(title='Model Input Feature'),
                height=450,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importances are not available for this model type.")
            
        st.subheader("Engine Diagnostic Specifications")
        st.markdown(f"""
        - **Selected Classifier**: `{best_model_name}`
        - **Target Metric Goal**: Premium Charges Billed in USD (`charges`)
        - **Splitting Parameters**: 80/20 train/test random partition validation
        - **Encoder Pipeline**: Scikit-Learn OneHotEncoder & ColumnTransformer
        - **Normalizing Scalar**: StandardScaler mapping Age and BMI to zero-mean and unit-variance
        """)
